"""Synthetic-tool transport for the opt-in ``工具抗截断/`` model prefix.

The design is adapted from Xeltra233/Antigravity-gateway (MIT). It operates on
the internal Gemini request/response shape so all public protocol converters can
reuse it without exposing the synthetic function call to clients.
"""

from __future__ import annotations

import copy
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from log import log

TOOL_ANTI_TRUNCATION_FLAG = "_tool_anti_truncation"
TOOL_NAME_PREFIX = "agw_emit_"
CONTENT_KEYS = ("content", "text", "answer", "response", "result", "output", "message")
NON_TEXT_MODEL_PATTERN = re.compile(
    r"(image|audio|tts|stt|whisper|embed|embedding|video|flux|dall-e|diffusion|moderation)",
    re.IGNORECASE,
)


@dataclass
class ToolAntiTruncationContext:
    tool_name: str
    stream_raw_buffers: dict[str, str] = field(default_factory=dict)
    stream_contents: dict[str, str] = field(default_factory=dict)
    stream_emitted: dict[str, int] = field(default_factory=dict)
    stream_synthetic_call_ids: set[str] = field(default_factory=set)
    stream_synthetic_choices: set[int] = field(default_factory=set)
    stream_real_tool_choices: set[int] = field(default_factory=set)


def _existing_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        declarations = tool.get("functionDeclarations") or tool.get("function_declarations")
        if isinstance(declarations, list):
            for declaration in declarations:
                if isinstance(declaration, dict) and declaration.get("name"):
                    names.add(str(declaration["name"]))
        custom = tool.get("custom")
        if isinstance(custom, dict) and custom.get("name"):
            names.add(str(custom["name"]))
    return names


def _generate_tool_name(existing: set[str]) -> str:
    while True:
        name = f"{TOOL_NAME_PREFIX}{secrets.token_hex(12)}"
        if name not in existing:
            return name


def _has_forced_real_tool(request: dict[str, Any], existing_tool_names: set[str]) -> bool:
    tool_config = request.get("toolConfig") or request.get("tool_config")
    if not isinstance(tool_config, dict):
        return False
    function_config = tool_config.get("functionCallingConfig") or tool_config.get(
        "function_calling_config"
    )
    if not isinstance(function_config, dict):
        return False
    mode = str(function_config.get("mode", "")).upper()
    allowed = function_config.get("allowedFunctionNames") or function_config.get(
        "allowed_function_names"
    )
    return mode == "NONE" or bool(allowed) or (mode == "ANY" and bool(existing_tool_names))


def _control_prompt(tool_name: str) -> str:
    return (
        f"Use the function `{tool_name}` exactly once for the final user-visible answer. "
        "Put the complete answer in its `content` string. Do not put ordinary answer text "
        "outside that function call. Preserve genuine client tool calls and never wrap them."
    )


def _append_control_message(request: dict[str, Any], tool_name: str) -> None:
    prompt = _control_prompt(tool_name)
    contents = request.get("contents")
    if not isinstance(contents, list):
        contents = []
    else:
        contents = copy.deepcopy(contents)

    last_role = ""
    for content in reversed(contents):
        if isinstance(content, dict) and content.get("role") in ("user", "model"):
            last_role = str(content["role"])
            break

    if not contents or last_role in ("", "model"):
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        request["contents"] = contents
        return

    system = request.get("systemInstruction") or request.get("system_instruction")
    if isinstance(system, str):
        system = {"parts": [{"text": system}]}
    elif isinstance(system, dict):
        system = copy.deepcopy(system)
    else:
        system = {"parts": []}
    parts = system.get("parts")
    if not isinstance(parts, list):
        parts = []
    parts.append({"text": prompt})
    system["parts"] = parts
    request.pop("system_instruction", None)
    request["systemInstruction"] = system


def prepare_tool_anti_truncation_request(
    body: dict[str, Any],
) -> tuple[dict[str, Any], ToolAntiTruncationContext | None]:
    """Consume the private feature flag and inject the synthetic internal tool."""
    if TOOL_ANTI_TRUNCATION_FLAG not in body:
        return body, None
    prepared = copy.deepcopy(body)
    enabled = bool(prepared.pop(TOOL_ANTI_TRUNCATION_FLAG, False))
    if not enabled:
        return prepared, None

    model_name = str(prepared.get("model", ""))
    request = prepared.get("request")
    if not isinstance(request, dict) or NON_TEXT_MODEL_PATTERN.search(model_name):
        return prepared, None
    tools = request.get("tools")
    existing_names = _existing_tool_names(tools)
    if _has_forced_real_tool(request, existing_names):
        log.info(f"工具抗截断透明回退：请求强制指定了真实工具，model={model_name}")
        return prepared, None

    tools = copy.deepcopy(tools) if isinstance(tools, list) else []
    tool_name = _generate_tool_name(existing_names)
    tools.append(
        {
            "functionDeclarations": [
                {
                    "name": tool_name,
                    "description": (
                        "Transport the complete final user-visible answer. Use exactly once; "
                        "do not wrap genuine tool calls."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    },
                }
            ]
        }
    )
    request["tools"] = tools
    tool_config = request.get("toolConfig")
    if not isinstance(tool_config, dict):
        tool_config = {}
    function_config = tool_config.get("functionCallingConfig")
    if not isinstance(function_config, dict):
        function_config = {"mode": "AUTO"}
    tool_config["functionCallingConfig"] = function_config
    request["toolConfig"] = tool_config
    _append_control_message(request, tool_name)
    prepared["request"] = request
    return prepared, ToolAntiTruncationContext(tool_name=tool_name)


def _decode_json_escape(char: str) -> str:
    return {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }.get(char, f"\\{char}")


def _scan_content_string(raw: str) -> str | None:
    for key in CONTENT_KEYS:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw)
        if not match:
            continue
        output: list[str] = []
        index = match.end()
        while index < len(raw):
            char = raw[index]
            if char == "\\":
                if index + 1 >= len(raw):
                    break
                escaped = raw[index + 1]
                if escaped == "u" and index + 5 < len(raw):
                    hex_value = raw[index + 2 : index + 6]
                    try:
                        codepoint = int(hex_value, 16)
                        if 0xD800 <= codepoint <= 0xDBFF:
                            if index + 11 >= len(raw) or raw[index + 6 : index + 8] != "\\u":
                                break
                            low = int(raw[index + 8 : index + 12], 16)
                            if not 0xDC00 <= low <= 0xDFFF:
                                break
                            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
                            index += 12
                        else:
                            index += 6
                        output.append(chr(codepoint))
                        continue
                    except ValueError:
                        pass
                output.append(_decode_json_escape(escaped))
                index += 2
                continue
            if char == '"':
                remainder = raw[index + 1 :].lstrip()
                if not remainder or remainder.startswith(("}", ",")):
                    return "".join(output)
                output.append(char)
                index += 1
                continue
            output.append(char)
            index += 1
        return "".join(output) if output else None
    return None


def extract_synthetic_content(arguments: Any) -> str | None:
    """Decode complete or deterministically repair partial synthetic arguments."""
    if isinstance(arguments, dict):
        for key in CONTENT_KEYS:
            value = arguments.get(key)
            if isinstance(value, str):
                return value
        return None
    if not isinstance(arguments, str):
        return None
    # Keep trailing whitespace: in a fragmented JSON string it may be part of
    # the user-visible content rather than insignificant JSON whitespace.
    raw = arguments.lstrip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1 :] if first_newline >= 0 else raw[3:]
        closing = raw.rfind("```")
        if closing >= 0:
            raw = raw[:closing]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _scan_content_string(raw)
    return extract_synthetic_content(parsed)


def _stream_content(
    context: ToolAntiTruncationContext,
    call_key: str,
    arguments: Any,
) -> str | None:
    if isinstance(arguments, dict):
        fragment = extract_synthetic_content(arguments)
        if fragment is None:
            return None
        previous = context.stream_contents.get(call_key, "")
        if fragment.startswith(previous):
            combined = fragment
        elif previous.startswith(fragment):
            combined = previous
        else:
            combined = previous + fragment
    elif isinstance(arguments, str):
        previous_raw = context.stream_raw_buffers.get(call_key, "")
        if arguments.startswith(previous_raw):
            combined_raw = arguments
        elif previous_raw.startswith(arguments):
            combined_raw = previous_raw
        else:
            combined_raw = previous_raw + arguments
        context.stream_raw_buffers[call_key] = combined_raw
        combined = extract_synthetic_content(combined_raw)
        if combined is None:
            return None
    else:
        return None

    context.stream_contents[call_key] = combined
    emitted = context.stream_emitted.get(call_key, 0)
    if len(combined) <= emitted:
        return None
    delta = combined[emitted:]
    context.stream_emitted[call_key] = len(combined)
    return delta


def transform_gemini_response(
    data: dict[str, Any],
    context: ToolAntiTruncationContext,
    *,
    streaming: bool = False,
) -> bool:
    """Replace the private function call with text while retaining genuine tools."""
    root = data.get("response") if isinstance(data.get("response"), dict) else data
    candidates = root.get("candidates") if isinstance(root, dict) else None
    if not isinstance(candidates, list):
        return False

    transformed = False
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        raw_choice_index = candidate.get("index", candidate_index)
        choice_index = raw_choice_index if isinstance(raw_choice_index, int) else candidate_index
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            parts = []

        synthetic_deltas: list[str] = []
        real_function_calls = (
            1 if streaming and choice_index in context.stream_real_tool_choices else 0
        )
        has_synthetic = streaming and choice_index in context.stream_synthetic_choices
        retained_parts: list[dict[str, Any]] = []
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            function_call = part.get("functionCall") or part.get("function_call")
            if isinstance(function_call, dict):
                name = str(function_call.get("name", ""))
                call_id = str(function_call.get("id", ""))
                if name == context.tool_name or (
                    streaming and call_id in context.stream_synthetic_call_ids
                ):
                    has_synthetic = True
                    if streaming:
                        context.stream_synthetic_choices.add(choice_index)
                        if call_id:
                            context.stream_synthetic_call_ids.add(call_id)
                    call_key = str(call_id or f"{choice_index}:{part_index}:{context.tool_name}")
                    arguments = function_call.get("args", function_call.get("arguments"))
                    decoded = (
                        _stream_content(context, call_key, arguments)
                        if streaming
                        else extract_synthetic_content(arguments)
                    )
                    if decoded:
                        synthetic_deltas.append(decoded)
                    continue
                real_function_calls += 1
                if streaming:
                    context.stream_real_tool_choices.add(choice_index)
            retained_parts.append(part)

        if not has_synthetic:
            continue
        transformed = True
        if isinstance(content, dict):
            retained_parts = [
                part for part in retained_parts if "text" not in part or part.get("thought") is True
            ]
            thought_parts = [part for part in retained_parts if part.get("thought") is True]
            other_parts = [part for part in retained_parts if part.get("thought") is not True]
            if synthetic_deltas:
                thought_parts.append({"text": "".join(synthetic_deltas)})
            content["role"] = "model"
            content["parts"] = thought_parts + other_parts
            candidate["content"] = content
        if real_function_calls == 0 and candidate.get("finishReason") in (
            "TOOL_CALLS",
            "tool_calls",
            "FUNCTION_CALL",
        ):
            candidate["finishReason"] = "STOP"
    return transformed


def transform_non_stream_response(
    content: bytes | str,
    context: ToolAntiTruncationContext | None,
) -> bytes | str:
    if context is None:
        return content
    was_bytes = isinstance(content, bytes)
    text = content.decode("utf-8", errors="replace") if was_bytes else content
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return content
    if not isinstance(data, dict) or not transform_gemini_response(data, context):
        return content
    result = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return result.encode("utf-8") if was_bytes else result


def transform_stream_item(
    item: bytes | str,
    context: ToolAntiTruncationContext | None,
) -> bytes | str:
    if context is None:
        return item
    was_bytes = isinstance(item, bytes)
    text = item.decode("utf-8", errors="replace") if was_bytes else item
    stripped = text.strip()
    if not stripped.startswith("data:") or stripped == "data: [DONE]":
        return item
    payload = stripped[5:].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return item
    if not isinstance(data, dict) or not transform_gemini_response(data, context, streaming=True):
        return item
    result = "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return result.encode("utf-8") if was_bytes else result
