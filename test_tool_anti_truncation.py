import copy
import json

import pytest
from fastapi import Response

import config
import src.api.antigravity as antigravity_api
import src.api.geminicli as geminicli_api
import src.quota_refresh as quota_refresh_module
import src.router.antigravity.model_list as antigravity_model_list
from src.converter.anti_truncation import AntiTruncationStreamProcessor, apply_anti_truncation
from src.models import OpenAIChatCompletionRequest
from src.router.antigravity.openai import chat_completions as antigravity_openai_chat
from src.router.geminicli.openai import chat_completions as geminicli_openai_chat
from src.converter.tool_anti_truncation import (
    TOOL_ANTI_TRUNCATION_FLAG,
    ToolAntiTruncationContext,
    _stream_content,
    extract_synthetic_content,
    prepare_tool_anti_truncation_request,
    transform_gemini_response,
    transform_non_stream_response,
    transform_stream_item,
)
from src.converter.antigravity_fix import normalize_antigravity_request
from src.utils import (
    get_available_models,
    get_base_model_from_feature_model,
    is_anti_truncation_model,
    is_tool_anti_truncation_model,
)


def test_stream_continuation_does_not_mutate_or_accumulate_into_original_history():
    original = {
        "model": "gemini-3.8-flash",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    }
    saved = copy.deepcopy(original)
    prepared = apply_anti_truncation(original)
    assert "tools" not in prepared["request"]
    assert "[done]" in prepared["request"]["systemInstruction"]["parts"][-1]["text"]

    processor = AntiTruncationStreamProcessor(lambda _: None, prepared)
    processor._append_content("first")
    processor.current_attempt = 2
    second = processor._build_current_payload()
    processor._append_content("second")
    processor.current_attempt = 3
    third = processor._build_current_payload()

    assert len(second["request"]["contents"]) == len(third["request"]["contents"]) == 3
    assert second["request"]["contents"][-2]["parts"][0]["text"] == "first"
    assert third["request"]["contents"][-2]["parts"][0]["text"] == "firstsecond"
    assert processor.base_payload == prepared
    assert original == saved


def test_tool_prefix_is_distinct_and_old_prefix_is_unchanged():
    tool_model = "工具抗截断/gemini-3.7-flash-low"
    old_model = "流式抗截断/gemini-3.7-flash-low"

    assert is_tool_anti_truncation_model(tool_model) is True
    assert is_anti_truncation_model(tool_model) is False
    assert is_anti_truncation_model(old_model) is True
    assert is_tool_anti_truncation_model(old_model) is False
    assert get_base_model_from_feature_model(tool_model) == "gemini-3.7-flash-low"
    models = get_available_models()
    assert any(model.startswith("工具抗截断/") for model in models)
    assert any(model.startswith("流式抗截断/") for model in models)


def test_normal_requests_keep_the_zero_copy_hot_path():
    body = {"model": "gemini-3.7-flash-low", "request": {"contents": []}}

    prepared, context = prepare_tool_anti_truncation_request(body)

    assert prepared is body
    assert context is None


@pytest.mark.asyncio
async def test_antigravity_dynamic_model_list_exposes_both_anti_truncation_modes(
    monkeypatch,
):
    async def models():
        return [{"id": "gemini-3.7-flash-low"}]

    monkeypatch.setattr(antigravity_model_list, "fetch_available_models", models)
    variants = await antigravity_model_list.get_antigravity_models_with_features()

    assert "流式抗截断/gemini-3.7-flash-low" in variants
    assert "工具抗截断/gemini-3.7-flash-low" in variants


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "api_module", "expected_model"),
    [
        (antigravity_openai_chat, antigravity_api, "gemini-3.7-flash-low"),
        (geminicli_openai_chat, geminicli_api, "gemini-3.7-flash"),
    ],
)
async def test_openai_routes_forward_tool_prefix_as_private_api_flag(
    monkeypatch, handler, api_module, expected_model
):
    captured = {}

    async def fake_non_stream_request(*, body, **_kwargs):
        captured.update(body)
        return Response(
            content=json.dumps(
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "ok"}],
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    }
                }
            ),
            status_code=200,
            media_type="application/json",
        )

    monkeypatch.setattr(api_module, "non_stream_request", fake_non_stream_request)
    request = OpenAIChatCompletionRequest(
        model="工具抗截断/gemini-3.7-flash-low",
        messages=[{"role": "user", "content": "hello"}],
    )

    response = await handler(request, token="test")

    assert response.status_code == 200
    assert captured["model"] == expected_model
    assert captured[TOOL_ANTI_TRUNCATION_FLAG] is True


def test_injection_preserves_real_tools_and_does_not_mutate_request():
    original = {
        "model": "gemini-3.7-flash-low",
        TOOL_ANTI_TRUNCATION_FLAG: True,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "search",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                }
            ],
        },
    }

    prepared, context = prepare_tool_anti_truncation_request(original)

    assert context is not None
    assert TOOL_ANTI_TRUNCATION_FLAG not in prepared
    assert TOOL_ANTI_TRUNCATION_FLAG in original
    names = [
        declaration["name"]
        for tool in prepared["request"]["tools"]
        for declaration in tool.get("functionDeclarations", [])
    ]
    assert names[0] == "search"
    assert context.tool_name in names
    assert len(context.tool_name.removeprefix("agw_emit_")) == 24
    system_text = prepared["request"]["systemInstruction"]["parts"][-1]["text"]
    assert context.tool_name in system_text


def test_trailing_model_turn_gets_a_user_control_turn():
    body = {
        "model": "gemini-2.5-flash",
        TOOL_ANTI_TRUNCATION_FLAG: True,
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": "start"}]},
                {"role": "model", "parts": [{"text": "prefill"}]},
            ]
        },
    }

    prepared, context = prepare_tool_anti_truncation_request(body)

    assert context is not None
    last = prepared["request"]["contents"][-1]
    assert last["role"] == "user"
    assert context.tool_name in last["parts"][0]["text"]


@pytest.mark.asyncio
async def test_tool_mode_preserves_prefill_while_normal_mode_keeps_old_behavior(
    monkeypatch,
):
    async def no_thoughts():
        return False

    monkeypatch.setattr(config, "get_return_thoughts_to_frontend", no_thoughts)
    request = {
        "model": "gemini-3.7-flash-low",
        "contents": [
            {"role": "user", "parts": [{"text": "start"}]},
            {"role": "model", "parts": [{"text": "prefill"}]},
        ],
    }

    normal = await normalize_antigravity_request(copy.deepcopy(request))
    preserved = await normalize_antigravity_request(
        copy.deepcopy(request),
        preserve_trailing_model=True,
    )

    assert normal["contents"][-1]["role"] == "user"
    assert preserved["contents"][-1]["role"] == "model"
    assert preserved["contents"][-1]["parts"][-1]["text"] == "prefill"


def test_forced_real_tool_transparently_bypasses_synthetic_transport():
    body = {
        "model": "claude-sonnet-4-6",
        TOOL_ANTI_TRUNCATION_FLAG: True,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "search"}]}],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": ["search"],
                }
            },
        },
    }

    prepared, context = prepare_tool_anti_truncation_request(body)

    assert context is None
    assert TOOL_ANTI_TRUNCATION_FLAG not in prepared
    assert "tools" not in prepared["request"]


def test_non_stream_response_restores_text_and_keeps_real_tool_calls():
    context = ToolAntiTruncationContext("agw_emit_test")
    response = {
        "response": {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "duplicate preamble"},
                            {"text": "reasoning", "thought": True},
                            {
                                "functionCall": {
                                    "name": "agw_emit_test",
                                    "args": {"content": "complete answer"},
                                }
                            },
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"q": "example"},
                                }
                            },
                        ],
                    },
                    "finishReason": "TOOL_CALLS",
                }
            ]
        }
    }

    encoded = json.dumps(response).encode()
    transformed = json.loads(transform_non_stream_response(encoded, context))
    candidate = transformed["response"]["candidates"][0]
    parts = candidate["content"]["parts"]

    assert {"text": "duplicate preamble"} not in parts
    assert {"text": "reasoning", "thought": True} in parts
    assert {"text": "complete answer"} in parts
    assert any(part.get("functionCall", {}).get("name") == "search" for part in parts)
    assert candidate["finishReason"] == "TOOL_CALLS"


def test_only_synthetic_tool_normalizes_finish_reason_to_stop():
    context = ToolAntiTruncationContext("agw_emit_test")
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "agw_emit_test",
                                "args": {"content": "answer"},
                            }
                        }
                    ]
                },
                "finishReason": "TOOL_CALLS",
            }
        ]
    }

    assert transform_gemini_response(response, context) is True
    assert response["candidates"][0]["finishReason"] == "STOP"


def test_partial_arguments_are_repaired_without_inventing_extra_text():
    raw = '{"content":"line one\\nline two with \\"quote\\"'

    assert extract_synthetic_content(raw) == 'line one\nline two with "quote"'


def test_streaming_argument_fragments_emit_only_new_content():
    context = ToolAntiTruncationContext("agw_emit_test")

    def event(arguments):
        return "data: " + json.dumps(
            {
                "response": {
                    "candidates": [
                        {
                            "index": 0,
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call-1",
                                            "name": "agw_emit_test",
                                            "args": arguments,
                                        }
                                    }
                                ]
                            },
                            "finishReason": None,
                        }
                    ]
                }
            }
        )

    first = transform_stream_item(event('{"content":"hello '), context)
    second = transform_stream_item(event('world"}'), context)
    first_data = json.loads(first[6:])
    second_data = json.loads(second[6:])

    first_text = first_data["response"]["candidates"][0]["content"]["parts"][0]["text"]
    second_text = second_data["response"]["candidates"][0]["content"]["parts"][0]["text"]
    assert first_text == "hello "
    assert second_text == "world"


def test_stream_tracks_synthetic_call_when_name_and_finish_arrive_separately():
    context = ToolAntiTruncationContext("agw_emit_test")
    first = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call-1",
                                "name": "agw_emit_test",
                                "args": '{"content":"hello',
                            }
                        }
                    ]
                },
            }
        ]
    }
    second = {
        "candidates": [
            {
                "index": 0,
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "call-1",
                                "args": ' world"}',
                            }
                        }
                    ]
                },
            }
        ]
    }
    final = {"candidates": [{"index": 0, "finishReason": "TOOL_CALLS"}]}

    assert transform_gemini_response(first, context, streaming=True) is True
    assert transform_gemini_response(second, context, streaming=True) is True
    assert transform_gemini_response(final, context, streaming=True) is True

    assert second["candidates"][0]["content"]["parts"] == [{"text": " world"}]
    assert final["candidates"][0]["finishReason"] == "STOP"


def test_stream_repairs_unicode_surrogate_split_across_argument_chunks():
    context = ToolAntiTruncationContext("agw_emit_test")

    assert _stream_content(context, "call-1", '{"content":"A\\uD83D') == "A"
    assert _stream_content(context, "call-1", '\\uDE00B"}') == "😀B"


@pytest.mark.asyncio
async def test_antigravity_api_injects_and_restores_synthetic_non_stream_response(
    monkeypatch,
):
    captured_tool_name = ""

    class FakeManager:
        async def get_valid_credential(self, **_kwargs):
            return "account.json", {
                "access_token": "token",
                "project_id": "project",
                "expiry": "2030-01-01T00:00:00+00:00",
            }

        async def record_api_call_result(self, *_args, **_kwargs):
            return None

    class FakeUpstreamResponse:
        status_code = 200
        headers = {"content-type": "application/json"}

        def __init__(self, content):
            self.content = content

    async def fake_post_async(*, json, **_kwargs):
        nonlocal captured_tool_name
        declarations = [
            declaration
            for tool in json["request"]["tools"]
            for declaration in tool.get("functionDeclarations", [])
        ]
        captured_tool_name = next(
            declaration["name"]
            for declaration in declarations
            if declaration["name"].startswith("agw_emit_")
        )
        return FakeUpstreamResponse(
            json_module.dumps(
                {
                    "response": {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": captured_tool_name,
                                                "args": {"content": "restored answer"},
                                            }
                                        }
                                    ]
                                },
                                "finishReason": "TOOL_CALLS",
                            }
                        ]
                    }
                }
            ).encode()
        )

    async def retry_config():
        return {"retry_enabled": True, "max_retries": 0, "retry_interval": 0}

    async def no_error_codes():
        return []

    async def stream_collection_disabled():
        return False

    async def no_quota_refresh(*_args, **_kwargs):
        return False

    json_module = json
    monkeypatch.setattr(antigravity_api, "credential_manager", FakeManager())
    monkeypatch.setattr(antigravity_api, "post_async", fake_post_async)
    monkeypatch.setattr(antigravity_api, "get_retry_config", retry_config)
    monkeypatch.setattr(antigravity_api, "get_auto_ban_error_codes", no_error_codes)
    monkeypatch.setattr(
        antigravity_api,
        "get_antigravity_stream2nostream",
        stream_collection_disabled,
    )
    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "maybe_schedule",
        no_quota_refresh,
    )

    response = await antigravity_api.non_stream_request(
        {
            "model": "gemini-3.7-flash-low",
            TOOL_ANTI_TRUNCATION_FLAG: True,
            "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        }
    )

    result = json.loads(response.body)
    parts = result["response"]["candidates"][0]["content"]["parts"]
    assert captured_tool_name.startswith("agw_emit_")
    assert parts == [{"text": "restored answer"}]
    assert result["response"]["candidates"][0]["finishReason"] == "STOP"


@pytest.mark.asyncio
async def test_antigravity_api_restores_synthetic_stream_before_protocol_conversion(
    monkeypatch,
):
    class FakeManager:
        async def get_valid_credential(self, **_kwargs):
            return "account.json", {
                "access_token": "token",
                "project_id": "project",
                "expiry": "2030-01-01T00:00:00+00:00",
            }

        async def record_api_call_result(self, *_args, **_kwargs):
            return None

    async def fake_stream_post_async(*, body, **_kwargs):
        tool_name = next(
            declaration["name"]
            for tool in body["request"]["tools"]
            for declaration in tool.get("functionDeclarations", [])
            if declaration["name"].startswith("agw_emit_")
        )
        yield "data: " + json.dumps(
            {
                "response": {
                    "candidates": [
                        {
                            "index": 0,
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call-1",
                                            "name": tool_name,
                                            "args": {"content": "streamed answer"},
                                        }
                                    }
                                ]
                            },
                            "finishReason": "TOOL_CALLS",
                        }
                    ]
                }
            }
        )
        yield "data: [DONE]"

    async def retry_config():
        return {"retry_enabled": True, "max_retries": 0, "retry_interval": 0}

    async def no_error_codes():
        return []

    async def no_quota_refresh(*_args, **_kwargs):
        return False

    monkeypatch.setattr(antigravity_api, "credential_manager", FakeManager())
    monkeypatch.setattr(antigravity_api, "stream_post_async", fake_stream_post_async)
    monkeypatch.setattr(antigravity_api, "get_retry_config", retry_config)
    monkeypatch.setattr(antigravity_api, "get_auto_ban_error_codes", no_error_codes)
    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "maybe_schedule",
        no_quota_refresh,
    )

    chunks = [
        chunk
        async for chunk in antigravity_api.stream_request(
            {
                "model": "gemini-3.7-flash-low",
                TOOL_ANTI_TRUNCATION_FLAG: True,
                "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            }
        )
    ]

    first = json.loads(chunks[0][6:])
    parts = first["response"]["candidates"][0]["content"]["parts"]
    assert parts == [{"text": "streamed answer"}]
    assert first["response"]["candidates"][0]["finishReason"] == "STOP"
    assert chunks[-1] == "data: [DONE]"
