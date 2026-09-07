import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import Response

from src.api import antigravity, geminicli

API_MODULES = (antigravity, geminicli)


def _image_result(*, applied: bool = True):
    primary = {
        "model": "gemini-2.5-flash",
        "request": {"variant": "image", "contents": []},
    }
    original = {
        "model": "gemini-2.5-flash",
        "request": {"variant": "text", "contents": []},
    }
    if not applied:
        primary = copy.deepcopy(original)
    return SimpleNamespace(
        payload=primary,
        original_payload=original,
        mode="all_text",
        applied=applied,
        converted_parts=1 if applied else 0,
        fallback_reason=None if applied else "local render fallback",
    )


def _install_variant_preparation(monkeypatch, api, *, applied=True):
    events = []

    def prepare_image(body):
        events.append(("image", body[api.IMAGE_CONTEXT_MODE_FLAG]))
        return _image_result(applied=applied)

    def prepare_tool(body):
        events.append(
            (
                "tool",
                body[api.TOOL_ANTI_TRUNCATION_FLAG],
                body["request"]["variant"],
            )
        )
        prepared = copy.deepcopy(body)
        prepared.pop(api.TOOL_ANTI_TRUNCATION_FLAG, None)
        return prepared, f'{body["request"]["variant"]}-context'

    monkeypatch.setattr(api, "prepare_image_context_request", prepare_image)
    monkeypatch.setattr(api, "prepare_tool_anti_truncation_request", prepare_tool)
    return events


async def _configure_api(monkeypatch, api, *, max_retries=0):
    credential = {"token": "token-a", "access_token": "token-a", "project_id": "project-a"}
    get_credential = AsyncMock(return_value=("cred-a.json", credential))
    monkeypatch.setattr(api.credential_manager, "get_valid_credential", get_credential)
    monkeypatch.setattr(
        api,
        "get_retry_config",
        AsyncMock(
            return_value={
                "retry_enabled": max_retries > 0,
                "max_retries": max_retries,
                "retry_interval": 0,
            }
        ),
    )
    monkeypatch.setattr(api, "get_auto_ban_error_codes", AsyncMock(return_value=[]))
    monkeypatch.setattr(api, "record_api_call_success", AsyncMock())
    monkeypatch.setattr(api, "record_api_call_error", AsyncMock())
    monkeypatch.setattr(api, "handle_error_with_retry", AsyncMock(return_value=False))
    if api is antigravity:
        monkeypatch.setattr(
            api, "get_antigravity_api_url", AsyncMock(return_value="https://upstream")
        )
        monkeypatch.setattr(api, "get_antigravity_stream2nostream", AsyncMock(return_value=False))
        monkeypatch.setattr(api, "get_confirmed_429_cooldown", AsyncMock(return_value=None))
    else:
        monkeypatch.setattr(
            api, "get_code_assist_endpoint", AsyncMock(return_value="https://upstream")
        )
        monkeypatch.setattr(api, "parse_and_log_cooldown", AsyncMock(return_value=None))
    return get_credential


@pytest.mark.parametrize("api", API_MODULES)
async def test_image_preparation_runs_before_tool_injection_and_auto_enables_it(monkeypatch, api):
    events = _install_variant_preparation(monkeypatch, api)

    prepared, context, fallback, fallback_context, available = await api._prepare_request_variants(
        {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "图片上下文/gemini"}
    )

    assert events == [
        ("image", "all_text"),
        ("tool", True, "image"),
        ("tool", True, "text"),
    ]
    assert prepared["model"] == "gemini-2.5-flash"
    assert context == "image-context"
    assert fallback["request"]["variant"] == "text"
    assert fallback_context == "text-context"
    assert available is True


@pytest.mark.parametrize("api", API_MODULES)
async def test_normal_request_skips_image_worker_and_preserves_fast_path(monkeypatch, api):
    body = {"model": "gemini-2.5-flash", "request": {}}
    image_prepare = AsyncMock(side_effect=AssertionError("image preparation must be skipped"))
    monkeypatch.setattr(api, "prepare_image_context_request", image_prepare)
    monkeypatch.setattr(
        api,
        "prepare_tool_anti_truncation_request",
        lambda value: (value, None),
    )

    prepared, context, fallback, fallback_context, available = await api._prepare_request_variants(
        body
    )

    assert prepared is body
    assert context is None
    assert fallback is None
    assert fallback_context is None
    assert available is False
    image_prepare.assert_not_called()


@pytest.mark.parametrize("api", API_MODULES)
async def test_local_image_fallback_still_auto_enables_tool_mode(monkeypatch, api):
    events = _install_variant_preparation(monkeypatch, api, applied=False)

    prepared, context, _, _, available = await api._prepare_request_variants(
        {api.IMAGE_CONTEXT_MODE_FLAG: "all_text"}
    )

    assert events[0] == ("image", "all_text")
    assert events[1:] == [("tool", True, "text")]
    assert prepared["request"]["variant"] == "text"
    assert context == "text-context"
    assert available is False


@pytest.mark.parametrize("api", API_MODULES)
async def test_non_stream_confirmed_multimodal_error_retries_text_on_same_credential(
    monkeypatch, api
):
    events = _install_variant_preparation(monkeypatch, api)
    get_credential = await _configure_api(monkeypatch, api, max_retries=0)
    calls = []

    async def post_async(*, url, json, headers):
        calls.append((copy.deepcopy(json), headers["Authorization"]))
        if len(calls) == 1:
            return httpx.Response(
                400,
                content=b'{"error":"unsupported inlineData format in request parts[0]"}',
            )
        return httpx.Response(200, content=b'{"response":{}}')

    transformed_contexts = []

    def transform(content, context):
        transformed_contexts.append(context)
        return b"transformed"

    monkeypatch.setattr(api, "post_async", post_async)
    monkeypatch.setattr(api, "transform_non_stream_response", transform)

    result = await api.non_stream_request(
        {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "图片上下文/gemini"}
    )

    assert result.status_code == 200
    assert result.body == b"transformed"
    assert [call[0]["request"]["variant"] for call in calls] == ["image", "text"]
    assert [call[0]["project"] for call in calls] == ["project-a", "project-a"]
    assert [call[1] for call in calls] == ["Bearer token-a", "Bearer token-a"]
    assert transformed_contexts == ["text-context"]
    get_credential.assert_awaited_once_with(
        mode="antigravity" if api is antigravity else "geminicli",
        model_name="gemini-2.5-flash",
    )
    api.record_api_call_error.assert_not_awaited()
    assert events[:3] == [
        ("image", "all_text"),
        ("tool", True, "image"),
        ("tool", True, "text"),
    ]


@pytest.mark.parametrize("api", API_MODULES)
async def test_stream_confirmed_multimodal_error_uses_fallback_synthetic_context(monkeypatch, api):
    _install_variant_preparation(monkeypatch, api)
    await _configure_api(monkeypatch, api, max_retries=0)
    calls = []

    async def stream_post_async(*, url, body, native, headers):
        calls.append((copy.deepcopy(body), headers["Authorization"]))
        if len(calls) == 1:
            yield Response(
                content='{"error":"inlineData is not supported for this image input"}',
                status_code=415,
            )
        else:
            yield "data: fallback"

    contexts = []

    def transform(chunk, context):
        contexts.append(context)
        return f"{chunk}:{context}"

    monkeypatch.setattr(api, "stream_post_async", stream_post_async)
    monkeypatch.setattr(api, "transform_stream_item", transform)

    output = [
        item
        async for item in api.stream_request(
            {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "图片上下文/gemini"}
        )
    ]

    assert output == ["data: fallback:text-context"]
    assert [call[0]["request"]["variant"] for call in calls] == ["image", "text"]
    assert [call[1] for call in calls] == ["Bearer token-a", "Bearer token-a"]
    assert contexts == ["text-context"]
    api.record_api_call_error.assert_not_awaited()


@pytest.mark.parametrize("api", API_MODULES)
async def test_stream_fallback_partial_output_is_not_retried(monkeypatch, api):
    _install_variant_preparation(monkeypatch, api)
    await _configure_api(monkeypatch, api, max_retries=2)
    calls = 0

    async def stream_post_async(*, url, body, native, headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield Response(
                content='{"error":"inlineData is not supported for this image input"}',
                status_code=415,
            )
            return
        yield "data: partial"
        raise RuntimeError("upstream disconnected")

    monkeypatch.setattr(api, "stream_post_async", stream_post_async)
    monkeypatch.setattr(
        api,
        "transform_stream_item",
        lambda chunk, context: f"{chunk}:{context}",
    )

    output = [
        item
        async for item in api.stream_request(
            {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "image-context/gemini"}
        )
    ]

    assert output[0] == "data: partial:text-context"
    assert isinstance(output[1], Response)
    assert output[1].status_code == 502
    assert "UPSTREAM_STREAM_INTERRUPTED" in output[1].body.decode()
    assert calls == 2
    api.record_api_call_error.assert_not_awaited()


@pytest.mark.parametrize("api", API_MODULES)
@pytest.mark.parametrize(
    ("status_code", "error_text"),
    [
        (400, "Invalid request parameter"),
        (429, "Unsupported inlineData image format due to quota"),
    ],
)
async def test_non_stream_generic_400_and_429_do_not_retry_text(
    monkeypatch, api, status_code, error_text
):
    _install_variant_preparation(monkeypatch, api)
    await _configure_api(monkeypatch, api, max_retries=0)
    calls = []

    async def post_async(*, url, json, headers):
        calls.append(copy.deepcopy(json))
        return httpx.Response(status_code, content=error_text.encode())

    monkeypatch.setattr(api, "post_async", post_async)

    result = await api.non_stream_request(
        {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "图片上下文/gemini"}
    )

    assert result.status_code == status_code
    assert len(calls) == 1
    assert calls[0]["request"]["variant"] == "image"
    api.record_api_call_error.assert_awaited_once()


@pytest.mark.parametrize("api", API_MODULES)
async def test_text_fallback_failure_resumes_credential_scheduling_with_updated_project(
    monkeypatch, api
):
    _install_variant_preparation(monkeypatch, api)
    await _configure_api(monkeypatch, api, max_retries=1)
    credentials = [
        (
            "cred-a.json",
            {"token": "token-a", "access_token": "token-a", "project_id": "project-a"},
        ),
        (
            "cred-b.json",
            {"token": "token-b", "access_token": "token-b", "project_id": "project-b"},
        ),
    ]
    api.credential_manager.get_valid_credential.side_effect = credentials
    api.handle_error_with_retry.return_value = True
    calls = []

    async def post_async(*, url, json, headers):
        calls.append((copy.deepcopy(json), headers["Authorization"]))
        if len(calls) == 1:
            return httpx.Response(
                400,
                content=b'{"error":"unsupported inlineData format in request parts[0]"}',
            )
        if len(calls) == 2:
            return httpx.Response(503, content=b'{"error":"temporarily unavailable"}')
        return httpx.Response(200, content=b'{"response":{}}')

    monkeypatch.setattr(api, "post_async", post_async)
    monkeypatch.setattr(api, "transform_non_stream_response", lambda content, context: content)

    result = await api.non_stream_request(
        {api.IMAGE_CONTEXT_MODE_FLAG: "all_text", "model": "图片上下文/gemini"}
    )

    assert result.status_code == 200
    assert [call[0]["request"]["variant"] for call in calls] == [
        "image",
        "text",
        "text",
    ]
    assert [call[0]["project"] for call in calls] == [
        "project-a",
        "project-a",
        "project-b",
    ]
    assert [call[1] for call in calls] == [
        "Bearer token-a",
        "Bearer token-a",
        "Bearer token-b",
    ]
    api.record_api_call_error.assert_awaited_once()


@pytest.mark.parametrize("api", API_MODULES)
@pytest.mark.parametrize(
    ("status_code", "error_text"),
    [
        (400, "Invalid request parameter"),
        (400, "Quota exceeded: invalid inlineData cannot be processed"),
        (429, "Unsupported inlineData image format due to quota"),
        (415, "Unsupported media type: application/json"),
        (400, "Invalid generation config at request.contents[0].parts[0]"),
    ],
)
def test_generic_or_quota_errors_do_not_trigger_image_fallback(api, status_code, error_text):
    assert api._is_confirmed_image_context_error(status_code, error_text) is False


@pytest.mark.parametrize("api", API_MODULES)
@pytest.mark.parametrize("status_code", [400, 415, 422])
def test_explicit_multimodal_shape_errors_trigger_image_fallback(api, status_code):
    assert (
        api._is_confirmed_image_context_error(
            status_code,
            'Invalid JSON payload: unknown name "inlineData" at request.contents[0].parts[0]',
        )
        is True
    )
