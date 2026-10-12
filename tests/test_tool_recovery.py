import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Response

import config
from src.api import antigravity, geminicli
from src.converter.image_context import IMAGE_CONTEXT_MODE_FLAG
from src.converter.tool_anti_truncation import TOOL_ANTI_TRUNCATION_FLAG


@pytest.fixture(params=[antigravity, geminicli], ids=["antigravity", "geminicli"])
def api(request, monkeypatch):
    module = request.param
    credential = {"access_token": "token", "token": "token", "project_id": "project"}
    monkeypatch.setattr(
        module.credential_manager,
        "get_valid_credential",
        AsyncMock(return_value=("test.json", credential)),
    )
    monkeypatch.setattr(
        module,
        "get_retry_config",
        AsyncMock(return_value={"retry_enabled": False, "max_retries": 0, "retry_interval": 0}),
    )
    monkeypatch.setattr(module, "get_auto_ban_error_codes", AsyncMock(return_value=[]))
    monkeypatch.setattr(module, "record_api_call_success", AsyncMock())
    monkeypatch.setattr(module, "record_api_call_error", AsyncMock())
    monkeypatch.setattr(config, "get_anti_truncation_max_attempts", AsyncMock(return_value=3))
    if module is antigravity:
        monkeypatch.setattr(
            module, "get_antigravity_api_url", AsyncMock(return_value="https://upstream")
        )
        monkeypatch.setattr(
            module, "get_antigravity_stream2nostream", AsyncMock(return_value=False)
        )
        monkeypatch.setattr(module, "get_confirmed_429_cooldown", AsyncMock(return_value=None))
    else:
        monkeypatch.setattr(
            module, "get_code_assist_endpoint", AsyncMock(return_value="https://upstream")
        )
        monkeypatch.setattr(module, "parse_and_log_cooldown", AsyncMock(return_value=None))
    return module


def _body():
    return {
        "model": "gemini-2.5-flash",
        TOOL_ANTI_TRUNCATION_FLAG: True,
        "request": {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    }


def _tool_name(body):
    return next(
        declaration["name"]
        for tool in body["request"]["tools"]
        for declaration in tool.get("functionDeclarations", [])
        if declaration["name"].startswith("agw_emit_")
    )


def _candidate(parts, reason=None, index=0):
    result = {"index": index, "content": {"role": "model", "parts": parts}}
    if reason:
        result["finishReason"] = reason
    return result


def _synthetic(body, content, reason="STOP", index=0):
    return _candidate(
        [{"functionCall": {"id": "private", "name": _tool_name(body), "args": content}}],
        reason,
        index,
    )


def _sse(*candidates):
    return "data: " + json.dumps({"response": {"candidates": candidates}})


def _candidates(output):
    for item in output:
        if isinstance(item, Response) or item.strip() == "data: [DONE]":
            continue
        data = json.loads(item.removeprefix("data: "))
        yield from data.get("response", data).get("candidates", [])


def _text(output, index=0):
    return "".join(
        part["text"]
        for candidate in _candidates(output)
        if candidate["index"] == index
        for part in candidate.get("content", {}).get("parts", [])
        if "text" in part and not part.get("thought")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_reason", "second_reason"),
    [
        ("MAX_TOKENS", "STOP"),
        ("STOP", "STOP"),
        ("MAX_TOKENS", None),
    ],
)
async def test_partial_synthetic_stream_continues_live_and_suppresses_replayed_prefix(
    api,
    monkeypatch,
    first_reason,
    second_reason,
):
    calls = []
    original = _body()
    saved = copy.deepcopy(original)

    async def upstream(*, body, **kwargs):
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            yield _sse(_synthetic(body, '{"content":"hello', first_reason))
        else:
            yield _sse(_synthetic(body, {"content": "hello world"}, second_reason))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    stream = api.stream_request(original)
    first = await anext(stream)
    assert _text([first]) == "hello"
    assert len(calls) == 1
    output = [first] + [item async for item in stream]
    assert _text(output) == "hello world"
    assert len(calls) == 2
    history = calls[1]["request"]["contents"]
    assert history[-2] == {"role": "model", "parts": [{"text": "hello"}]}
    assert history[-1]["role"] == "user"
    assert [
        candidate["finishReason"]
        for candidate in _candidates(output)
        if "finishReason" in candidate
    ] == ["STOP"]
    assert output.count("data: [DONE]") == 1
    assert original == saved


@pytest.mark.asyncio
async def test_disconnected_plain_text_stream_resumes_from_emitted_answer(api, monkeypatch):
    calls = []

    async def upstream(*, body, **kwargs):
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            yield _sse(_candidate([{"text": "first"}]))
            raise RuntimeError("connection lost")
        yield _sse(_synthetic(body, {"content": " second"}))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert _text(output) == "first second"
    assert len(calls) == 2
    assert calls[1]["request"]["contents"][-2]["parts"][0]["text"] == "first"
    assert not any(isinstance(item, Response) for item in output)


@pytest.mark.asyncio
@pytest.mark.parametrize("real_tool", [False, True])
async def test_plain_stop_and_genuine_tool_handoff_do_not_trigger_continuation(
    api, monkeypatch, real_tool
):
    calls = 0
    part = (
        {"functionCall": {"id": "real", "name": "read", "args": {"path": "file"}}}
        if real_tool
        else {"text": "complete"}
    )

    async def upstream(**kwargs):
        nonlocal calls
        calls += 1
        yield _sse(_candidate([part], "TOOL_CALLS" if real_tool else "STOP"))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert calls == 1
    assert list(_candidates(output))[0]["content"]["parts"] == [part]


@pytest.mark.asyncio
async def test_exhausted_recovery_reports_error_without_fake_success(api, monkeypatch):
    calls = 0

    async def upstream(*, body, **kwargs):
        nonlocal calls
        calls += 1
        yield _sse(_synthetic(body, {"content": "partial"}, "MAX_TOKENS"))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert calls == 3
    assert isinstance(output[-1], Response)
    assert output[-1].status_code == 502
    assert "TOOL_ANTI_TRUNCATION_INCOMPLETE" in output[-1].body.decode()
    assert not any(candidate.get("finishReason") == "STOP" for candidate in _candidates(output))


@pytest.mark.asyncio
async def test_http_error_before_output_is_left_to_existing_credential_retries(api, monkeypatch):
    calls = 0

    async def upstream(**kwargs):
        nonlocal calls
        calls += 1
        yield Response('{"error":"rate limit"}', status_code=429)

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert calls == 1
    assert len(output) == 1 and output[0].status_code == 429


@pytest.mark.asyncio
async def test_non_stream_recovers_and_merges_answer_and_usage(api, monkeypatch):
    calls = []
    json_module = json

    async def post(*, json: dict, **kwargs):
        calls.append(copy.deepcopy(json))
        candidate = _synthetic(
            json,
            {"content": "first" if len(calls) == 1 else " second"},
            "MAX_TOKENS" if len(calls) == 1 else "STOP",
        )
        return httpx.Response(
            200,
            content=json_module.dumps(
                {"response": {"candidates": [candidate], "usageMetadata": {"totalTokenCount": 10}}}
            ).encode(),
        )

    monkeypatch.setattr(api, "post_async", post)
    response = await api.non_stream_request(_body())
    assert response.status_code == 200
    root = json.loads(response.body)["response"]
    assert (
        "".join(part.get("text", "") for part in root["candidates"][0]["content"]["parts"])
        == "first second"
    )
    assert root["candidates"][0]["finishReason"] == "STOP"
    assert root["usageMetadata"]["totalTokenCount"] == 20
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("complete_text", "partial_text"), [("A", "B"), ("same", "same")])
async def test_recovery_only_reissues_unfinished_candidate(
    api, monkeypatch, complete_text, partial_text
):
    calls = []

    async def upstream(*, body, **kwargs):
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            yield _sse(
                _synthetic(body, {"content": complete_text}, index=0),
                _synthetic(body, {"content": partial_text}, "MAX_TOKENS", index=1),
            )
        else:
            yield _sse(_synthetic(body, {"content": " rest"}))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert _text(output, 0) == complete_text
    assert _text(output, 1) == partial_text + " rest"
    assert calls[1]["request"]["generationConfig"]["candidateCount"] == 1
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_image_rejection_is_not_repeated_during_tool_continuation(api, monkeypatch):
    preparations = []
    calls = []

    def image(body):
        preparations.append(1)
        original = copy.deepcopy(body)
        original.pop(IMAGE_CONTEXT_MODE_FLAG, None)
        prepared = copy.deepcopy(original)
        prepared["request"]["representation"] = "image"
        original["request"]["representation"] = "text"
        return SimpleNamespace(
            payload=prepared, original_payload=original, mode="all_text", applied=True
        )

    async def upstream(*, body, **kwargs):
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            yield Response('{"error":"inlineData image input is not supported"}', status_code=415)
            return
        yield _sse(
            _synthetic(
                body,
                {"content": "first" if len(calls) == 2 else " second"},
                "MAX_TOKENS" if len(calls) == 2 else "STOP",
            )
        )
        yield "data: [DONE]"

    monkeypatch.setattr(api, "prepare_image_context_request", image)
    monkeypatch.setattr(api, "stream_post_async", upstream)
    body = _body()
    body.pop(TOOL_ANTI_TRUNCATION_FLAG)
    body[IMAGE_CONTEXT_MODE_FLAG] = "all_text"
    output = [item async for item in api.stream_request(body)]
    assert _text(output) == "first second"
    assert len(calls) == 3 and len(preparations) == 1
    assert [call["request"].get("representation") for call in calls] == ["image", "text", None]


@pytest.mark.asyncio
async def test_partial_genuine_tool_is_not_replayed_after_connection_loss(api, monkeypatch):
    calls = 0

    async def upstream(**kwargs):
        nonlocal calls
        calls += 1
        yield _sse(_candidate([{"functionCall": {"id": "real", "name": "write", "args": {}}}]))
        raise RuntimeError("connection lost")

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert calls == 1
    assert isinstance(output[-1], Response) and output[-1].status_code == 502


@pytest.mark.asyncio
async def test_partial_429_keeps_quota_error_recording_during_recovery(api, monkeypatch):
    calls = 0
    if api is antigravity:
        api.get_confirmed_429_cooldown.side_effect = RuntimeError("quota endpoint unavailable")

    async def upstream(*, body, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield _sse(_candidate([{"text": "first"}]))
            yield Response('{"error":{"code":429,"message":"rate limit"}}', status_code=429)
            return
        yield _sse(_synthetic(body, {"content": " second"}))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert _text(output) == "first second" and calls == 2
    api.record_api_call_error.assert_awaited_once()
    assert api.record_api_call_error.await_args.args[2] == 429
    if api is antigravity:
        api.get_confirmed_429_cooldown.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_tool_stream_emits_framed_bytes(api, monkeypatch):
    async def upstream(*, body, **kwargs):
        yield _sse(_synthetic(body, {"content": "complete"}))
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body(), native=True)]
    assert all(isinstance(item, bytes) and item.endswith(b"\n\n") for item in output)
    assert _text([item.decode().strip() for item in output]) == "complete"


@pytest.mark.asyncio
async def test_short_continuation_is_not_lost_when_it_resembles_previous_answer(api, monkeypatch):
    calls = 0

    async def upstream(*, body, **kwargs):
        nonlocal calls
        calls += 1
        yield _sse(
            _synthetic(
                body,
                {"content": "hello" if calls == 1 else "he"},
                "MAX_TOKENS" if calls == 1 else "STOP",
            )
        )
        yield "data: [DONE]"

    monkeypatch.setattr(api, "stream_post_async", upstream)
    output = [item async for item in api.stream_request(_body())]
    assert _text(output) == "hellohe"
    assert calls == 2
