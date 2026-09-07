import json

import pytest
from fastapi import Response

import src.api.antigravity as antigravity_api
import src.api.geminicli as geminicli_api
from src.converter.image_context import IMAGE_CONTEXT_MODE_FLAG, ImageContextMode
from src.models import ClaudeRequest, GeminiRequest, OpenAIChatCompletionRequest
from src.router.antigravity.anthropic import messages as antigravity_anthropic
from src.router.antigravity.gemini import generate_content as antigravity_gemini
from src.router.antigravity.openai import chat_completions as antigravity_openai
from src.router.geminicli.anthropic import messages as geminicli_anthropic
from src.router.geminicli.gemini import generate_content as geminicli_gemini
from src.router.geminicli.openai import chat_completions as geminicli_openai


def _success_response() -> Response:
    return Response(
        content=json.dumps(
            {
                "response": {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {},
                }
            }
        ),
        status_code=200,
        media_type="application/json",
    )


@pytest.mark.asyncio
async def test_all_public_routes_forward_mixed_image_context_as_one_private_flag(
    monkeypatch,
):
    captured = []

    async def fake_non_stream_request(*, body, **_kwargs):
        captured.append(body)
        return _success_response()

    monkeypatch.setattr(antigravity_api, "non_stream_request", fake_non_stream_request)
    monkeypatch.setattr(geminicli_api, "non_stream_request", fake_non_stream_request)

    model = "[混合实验性] gemini-2.5-flash"
    openai_request = OpenAIChatCompletionRequest(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
    )
    anthropic_request = ClaudeRequest(
        model=model,
        max_tokens=128,
        messages=[{"role": "user", "content": "hello"}],
    )
    gemini_request = GeminiRequest(contents=[{"role": "user", "parts": [{"text": "hello"}]}])

    await antigravity_openai(openai_request, token="test")
    await antigravity_anthropic(anthropic_request, _token="test")
    await antigravity_gemini(gemini_request, model=model, api_key="test")
    await geminicli_openai(openai_request, token="test")
    await geminicli_anthropic(anthropic_request, token="test")
    await geminicli_gemini(gemini_request, model=model, api_key="test")

    assert len(captured) == 6
    for body in captured:
        assert body["model"] == "gemini-2.5-flash"
        assert body[IMAGE_CONTEXT_MODE_FLAG] == ImageContextMode.CURRENT_TURN.value


@pytest.mark.asyncio
async def test_full_image_marker_uses_all_text_mode(monkeypatch):
    captured = {}

    async def fake_non_stream_request(*, body, **_kwargs):
        captured.update(body)
        return _success_response()

    monkeypatch.setattr(geminicli_api, "non_stream_request", fake_non_stream_request)
    request = OpenAIChatCompletionRequest(
        model="[实验性]gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    await geminicli_openai(request, token="test")

    assert captured["model"] == "gemini-2.5-flash"
    assert captured[IMAGE_CONTEXT_MODE_FLAG] == ImageContextMode.ALL_TEXT.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "api_module", "auth_keyword"),
    [
        (antigravity_anthropic, antigravity_api, "_token"),
        (geminicli_anthropic, geminicli_api, "token"),
    ],
)
async def test_anthropic_stream_exposes_midstream_upstream_failure(
    monkeypatch, handler, api_module, auth_keyword
):
    async def interrupted_stream_request(*, body, **_kwargs):
        yield (
            b'data: {"response":{"candidates":[{"content":{"role":"model",'
            b'"parts":[{"text":"partial"}]}}]}}\n\n'
        )
        yield Response(
            content=json.dumps(
                {
                    "error": {
                        "code": 502,
                        "message": "UPSTREAM_STREAM_INTERRUPTED",
                        "status": "UPSTREAM_STREAM_INTERRUPTED",
                    }
                }
            ),
            status_code=502,
            media_type="application/json",
        )

    monkeypatch.setattr(api_module, "stream_request", interrupted_stream_request)
    request = ClaudeRequest(
        model="gemini-2.5-flash",
        max_tokens=128,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    response = await handler(request, **{auth_keyword: "test"})
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
    payload = b"".join(chunks).decode("utf-8")

    assert "event: error" in payload
    assert "UPSTREAM_STREAM_INTERRUPTED" in payload
    assert "message_stop" not in payload
    assert '"stop_reason":"end_turn"' not in payload
