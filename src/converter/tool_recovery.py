"""Continuation and stream recovery for synthetic-tool answer transport.

Works after the synthetic call has been restored to Gemini text, before any
public protocol conversion. Credential retries remain in the API clients.
"""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from fastapi import Response

from log import log
from src.converter.image_context import IMAGE_CONTEXT_MODE_FLAG
from src.converter.tool_anti_truncation import (
    TOOL_ANTI_TRUNCATION_FLAG,
    ToolAntiTruncationContext,
)


def tool_recovery_enabled(body: dict[str, Any]) -> bool:
    return bool(body.get(TOOL_ANTI_TRUNCATION_FLAG) or body.get(IMAGE_CONTEXT_MODE_FLAG))


@dataclass
class _Choice:
    text: list[str] = field(default_factory=list)
    parts: list[dict[str, Any]] = field(default_factory=list)
    candidate: dict[str, Any] = field(default_factory=dict)
    complete: bool = False
    finish_sent: bool = False
    real_tools: bool = False
    native_output: bool = False
    finish_reason: str | None = None
    repeat_prefix: str | None = None
    pending_text: str = ""


class ToolAntiTruncationRecovery:
    def __init__(self, body: dict[str, Any], max_attempts: int):
        self.body = copy.deepcopy(body)
        self.max_attempts = max_attempts
        # The API client updates this when switching image/text representations.
        self.context: ToolAntiTruncationContext | None = None
        self.choices: dict[int, _Choice] = {}
        self.metadata: dict[str, Any] = {}
        self.outer_metadata: dict[str, Any] = {}
        self.wrapped = False
        self.usage: dict[str, Any] = {}
        self.attempt_usage: dict[str, Any] = {}
        self.text_fallback = False

    def _continuation(self, index: int) -> dict[str, Any]:
        body = copy.deepcopy(self.body)
        if self.text_fallback:
            body.pop(IMAGE_CONTEXT_MODE_FLAG, None)
            body[TOOL_ANTI_TRUNCATION_FLAG] = True
        request = body.setdefault("request", {})
        contents = list(request.get("contents") or [])
        text = "".join(self.choices[index].text)
        if text:
            parts = [{"text": text}]
            if contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "model":
                contents[-1] = {
                    **contents[-1],
                    "parts": list(contents[-1].get("parts") or []) + parts,
                }
            else:
                contents.append({"role": "model", "parts": parts})
        continuation = {
            "text": (
                "The previous answer was interrupted. Continue exactly where it stopped. "
                "Do not repeat any text already provided, add a preface, or restart the answer. "
                "Use the answer transport function for the remaining answer. "
                "Preserve genuine client tool calls."
            )
        }
        if contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "user":
            contents[-1] = {
                **contents[-1],
                "parts": list(contents[-1].get("parts") or []) + [continuation],
            }
        else:
            contents.append({"role": "user", "parts": [continuation]})
        request["contents"] = contents
        generation = dict(request.get("generationConfig") or {})
        generation["candidateCount"] = 1
        request["generationConfig"] = generation
        return body

    @staticmethod
    def _new_text(choice: _Choice, text: str) -> str:
        if choice.repeat_prefix is None:
            return text
        choice.pending_text += text
        prefix = choice.repeat_prefix
        if choice.pending_text.startswith(prefix):
            result = choice.pending_text[len(prefix) :]
        elif prefix.startswith(choice.pending_text):
            return ""
        else:
            result = choice.pending_text
        choice.repeat_prefix = None
        choice.pending_text = ""
        return result

    @staticmethod
    def _flush_text(choice: _Choice) -> str:
        # A short continuation may merely resemble the start of the previous
        # answer. Suppress only a confirmed full-prefix replay, never lose
        # ambiguous text when a response ends or disconnects.
        text = choice.pending_text
        choice.pending_text = ""
        choice.repeat_prefix = None
        if text:
            choice.text.append(text)
            choice.parts.append({"text": text})
        return text

    def _usage_with_attempt(self) -> dict[str, Any]:
        result = dict(self.usage)
        for key, value in self.attempt_usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = result.get(key, 0) + value
            else:
                result[key] = value
        return result

    def _wrap(self, root: dict[str, Any]) -> dict[str, Any]:
        return {**self.outer_metadata, "response": root} if self.wrapped else root

    def _process(self, data: dict[str, Any], target: int | None, observed: set[int]):
        root = data.get("response") if isinstance(data.get("response"), dict) else data
        self.wrapped = root is not data
        if self.wrapped:
            self.outer_metadata.update(
                {key: value for key, value in data.items() if key != "response"}
            )
        self.metadata.update(
            {
                key: value
                for key, value in root.items()
                if key not in ("candidates", "usageMetadata")
            }
        )
        if isinstance(root.get("usageMetadata"), dict):
            self.attempt_usage.update(root["usageMetadata"])
            root["usageMetadata"] = self._usage_with_attempt()

        for position, candidate in enumerate(root.get("candidates") or []):
            raw_index = candidate.get("index", position)
            if not isinstance(raw_index, int):
                raw_index = position
            index = target if target is not None else raw_index
            candidate["index"] = index
            observed.add(index)
            choice = self.choices.setdefault(index, _Choice())
            parts = (candidate.get("content") or {}).get("parts") or []
            reason = candidate.get("finishReason")
            output_parts = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str) and not part.get("thought"):
                    text = self._new_text(choice, part["text"])
                    if not text:
                        continue
                    part = {**part, "text": text}
                    choice.text.append(text)
                output_parts.append(part)
                choice.parts.append(copy.deepcopy(part))
                if "functionCall" in part or "function_call" in part:
                    choice.real_tools = True
                if any(
                    key in part
                    for key in ("inlineData", "fileData", "executableCode", "codeExecutionResult")
                ):
                    choice.native_output = True
            if reason and choice.pending_text:
                output_parts.append({"text": self._flush_text(choice)})
            if isinstance(candidate.get("content"), dict):
                candidate["content"]["parts"] = output_parts
            choice.candidate.update(copy.deepcopy(candidate))
            if reason:
                choice.finish_reason = reason
                synthetic = raw_index in self.context.stream_synthetic_choices
                arguments_complete = raw_index in self.context.stream_complete_synthetic_choices
                if reason in ("STOP", "TOOL_CALLS", "tool_calls", "FUNCTION_CALL"):
                    choice.complete = bool(
                        choice.text or choice.real_tools or choice.native_output
                    ) and (not synthetic or arguments_complete or choice.real_tools)
                elif reason not in ("MAX_TOKENS", "LENGTH", "length"):
                    # Preserve terminal upstream reasons such as SAFETY rather
                    # than disguising them as a transport interruption.
                    choice.complete = True
                if choice.complete:
                    choice.finish_sent = True
                else:
                    candidate.pop("finishReason", None)
        return data

    @staticmethod
    def _encode(data: dict[str, Any]) -> str:
        return "data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _incomplete_error() -> Response:
        return Response(
            json.dumps(
                {
                    "error": {
                        "code": 502,
                        "message": "工具抗截断续写次数已用尽，回答仍未完整结束",
                        "status": "TOOL_ANTI_TRUNCATION_INCOMPLETE",
                    }
                },
                ensure_ascii=False,
            ),
            status_code=502,
            media_type="application/json",
        )

    async def process_stream(self, request: Callable) -> AsyncIterator[str | bytes | Response]:
        pending = deque([(None, 1)])
        while pending:
            target, attempt = pending.popleft()
            payload = self.body if target is None else self._continuation(target)
            if target is not None:
                choice = self.choices[target]
                choice.finish_reason = None
                choice.repeat_prefix = "".join(choice.text)
                choice.pending_text = ""
            self.context = None
            self.attempt_usage = {}
            observed: set[int] = set()
            error = None
            done = False
            stream = request(payload)
            try:
                async for item in stream:
                    if not isinstance(self.context, ToolAntiTruncationContext):
                        # Forced tool_choice and non-text models retain the
                        # original API behavior, including its error handling.
                        yield item
                        continue
                    if isinstance(item, Response):
                        error = item
                        break
                    text = (
                        item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
                    )
                    stripped = text.strip()
                    if stripped == "data: [DONE]":
                        done = True
                        break
                    if not stripped.startswith("data:"):
                        yield item
                        continue
                    try:
                        data = json.loads(stripped[5:].strip())
                    except (ValueError, TypeError):
                        yield item
                        continue
                    if not isinstance(data, dict):
                        yield item
                        continue
                    if "error" in data:
                        error = Response(
                            json.dumps(data), status_code=502, media_type="application/json"
                        )
                        break
                    yield self._encode(self._process(data, target, observed))
            except Exception as exc:
                error = Response(
                    json.dumps(
                        {
                            "error": {
                                "code": 502,
                                "message": str(exc),
                                "status": "UPSTREAM_STREAM_INTERRUPTED",
                            }
                        }
                    ),
                    status_code=502,
                    media_type="application/json",
                )
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

            if not isinstance(self.context, ToolAntiTruncationContext):
                if error is not None:
                    yield error
                return
            self.usage = self._usage_with_attempt()
            unfinished = []
            for index in observed:
                choice = self.choices[index]
                pending_text = self._flush_text(choice)
                if pending_text:
                    yield self._encode(
                        self._wrap(
                            {
                                "candidates": [
                                    {
                                        "index": index,
                                        "content": {
                                            "role": "model",
                                            "parts": [{"text": pending_text}],
                                        },
                                    }
                                ]
                            }
                        )
                    )
                raw_index = 0 if target is not None else index
                if (
                    done
                    and raw_index in self.context.stream_complete_synthetic_choices
                    and choice.text
                ):
                    if not choice.finish_reason:
                        choice.complete = True
                if (choice.real_tools or choice.native_output) and error is None:
                    # Hand execution back to the client; never ask the model
                    # to repeat an actual tool call during continuation.
                    choice.complete = True
                if choice.complete:
                    if not choice.finish_sent:
                        reason = choice.finish_reason or (
                            "TOOL_CALLS" if choice.real_tools else "STOP"
                        )
                        yield self._encode(
                            self._wrap(
                                {
                                    "candidates": [
                                        {
                                            "index": index,
                                            "content": {"role": "model", "parts": []},
                                            "finishReason": reason,
                                        }
                                    ]
                                }
                            )
                        )
                        choice.finish_sent = True
                        choice.finish_reason = reason
                    continue
                unfinished.append(index)

            if error is not None and (
                not observed
                or any(
                    self.choices[i].real_tools or self.choices[i].native_output for i in observed
                )
            ):
                yield error
                return
            if error is not None and not (
                error.status_code >= 500 or error.status_code in (408, 429)
            ):
                yield error
                return
            if unfinished or not observed:
                if attempt >= self.max_attempts:
                    yield error or self._incomplete_error()
                    return
                if not observed:
                    index = target if target is not None else 0
                    self.choices.setdefault(index, _Choice())
                    unfinished = [index]
                for index in unfinished:
                    log.info(
                        f"工具抗截断续写：candidate={index}, attempt={attempt + 1}/{self.max_attempts}"
                    )
                    pending.append((index, attempt + 1))
        yield "data: [DONE]"

    async def process_non_stream(self, request: Callable) -> Response:
        last_response = None

        async def as_stream(payload):
            nonlocal last_response
            last_response = await request(payload)
            if last_response.status_code != 200:
                yield last_response
                return
            if not isinstance(self.context, ToolAntiTruncationContext):
                return
            yield "data: " + last_response.body.decode("utf-8")
            yield "data: [DONE]"

        async for item in self.process_stream(as_stream):
            if isinstance(item, Response):
                return item
        if not self.choices:
            return last_response
        root = dict(self.metadata)
        root["candidates"] = [
            {
                **choice.candidate,
                "index": index,
                "content": {"role": "model", "parts": choice.parts},
                "finishReason": choice.finish_reason
                or ("TOOL_CALLS" if choice.real_tools else "STOP"),
            }
            for index, choice in sorted(self.choices.items())
        ]
        if self.usage:
            root["usageMetadata"] = self.usage
        headers = dict(last_response.headers)
        headers.pop("content-length", None)
        headers.pop("content-encoding", None)
        return Response(
            json.dumps(self._wrap(root), ensure_ascii=False), status_code=200, headers=headers
        )
