from datetime import datetime

import pytest
from fastapi import Response

import src.api.antigravity as antigravity_api
import src.api.utils as api_utils
import src.quota_refresh as quota_refresh_module
from src.api.antigravity import _is_retryable_status
from src.credential_manager import (
    ANTIGRAVITY_CLAUDE_POOL_KEY,
    ANTIGRAVITY_GEMINI_POOL_KEY,
    CredentialManager,
    normalize_model_cooldown_key,
)


def test_antigravity_quota_reset_timestamp_is_not_discarded():
    reset_at = "2030-09-01T15:40:16Z"
    error = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "QUOTA_EXHAUSTED",
                    "metadata": {
                        "quotaResetDelay": "155h4m55.216292832s",
                        "quotaResetTimeStamp": reset_at,
                    },
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "558295.216292832s",
                },
            ],
        }
    }

    parsed = api_utils.parse_quota_reset_timestamp(error, mode="antigravity")
    expected = datetime.fromisoformat(reset_at.replace("Z", "+00:00")).timestamp()

    assert parsed == pytest.approx(expected, abs=1)


def test_retry_delay_is_used_when_reset_timestamp_is_missing(monkeypatch):
    monkeypatch.setattr(api_utils.time, "time", lambda: 1_000.0)
    error = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "1h2m3.5s",
                }
            ],
        }
    }

    assert api_utils.parse_quota_reset_timestamp(error, mode="antigravity") == 4723.5


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("gemini-3.7-flash-low", ANTIGRAVITY_GEMINI_POOL_KEY),
        ("gemini-3-pro-high", ANTIGRAVITY_GEMINI_POOL_KEY),
        ("claude-sonnet-4-5", ANTIGRAVITY_CLAUDE_POOL_KEY),
    ],
)
def test_antigravity_models_share_two_cooldown_pools(model_name, expected):
    assert normalize_model_cooldown_key("antigravity", model_name) == expected
    assert normalize_model_cooldown_key("geminicli", model_name) == model_name


@pytest.mark.asyncio
async def test_credential_manager_selects_and_records_by_antigravity_pool():
    calls = []

    class FakeBackend:
        async def get_next_available_credential(self, *, mode, model_name):
            calls.append(("select", mode, model_name))
            return None

        async def set_model_cooldown(self, filename, model_name, cooldown_until, *, mode):
            calls.append(("cooldown", filename, model_name, cooldown_until, mode))

    class FakeStorage:
        def __init__(self):
            self._backend = FakeBackend()

        async def update_credential_state(self, filename, updates, *, mode):
            calls.append(("state", filename, updates, mode))
            return True

    manager = CredentialManager()
    manager._initialized = True
    manager._storage_adapter = FakeStorage()

    assert (
        await manager.get_valid_credential(mode="antigravity", model_name="gemini-3-pro-high")
        is None
    )
    await manager.record_api_call_result(
        "account.json",
        False,
        429,
        cooldown_until=2_000_000_000.0,
        mode="antigravity",
        model_name="gemini-3.7-flash-low",
    )

    assert ("select", "antigravity", ANTIGRAVITY_GEMINI_POOL_KEY) in calls
    assert (
        "cooldown",
        "account.json",
        ANTIGRAVITY_GEMINI_POOL_KEY,
        2_000_000_000.0,
        "antigravity",
    ) in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status", [403, 429])
@pytest.mark.parametrize("streaming", [False, True])
async def test_retry_records_unavailable_credential_before_selecting_next(
    monkeypatch, first_status, streaming
):
    class FakeResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content
            self.text = content.decode("utf-8")
            self.headers = {"content-type": "application/json"}

    class FakeManager:
        def __init__(self):
            self.selection_count = 0
            self.error_recorded = False
            self.disabled = False

        async def get_valid_credential(self, *, mode, model_name):
            self.selection_count += 1
            if self.selection_count == 1:
                return "bad.json", {
                    "access_token": "bad",
                    "project_id": "bad-project",
                }
            assert self.error_recorded is True
            if first_status == 403:
                assert self.disabled is True
            return "good.json", {
                "access_token": "good",
                "project_id": "good-project",
            }

        async def record_api_call_result(
            self,
            _filename,
            success,
            _error_code=None,
            **_kwargs,
        ):
            if not success:
                self.error_recorded = True

        async def set_cred_disabled(self, _filename, disabled, *, mode):
            assert mode == "antigravity"
            self.disabled = disabled
            return True

    manager = FakeManager()
    responses = [
        FakeResponse(
            first_status,
            (
                b'{"error":{"status":"RESOURCE_EXHAUSTED","details":' b'[{"retryDelay":"1h"}]}}'
                if first_status == 429
                else b'{"error":{"status":"PERMISSION_DENIED"}}'
            ),
        ),
        FakeResponse(200, b'{"ok":true}'),
    ]

    async def fake_post_async(**_kwargs):
        return responses.pop(0)

    async def fake_stream_post_async(**_kwargs):
        response = responses.pop(0)
        if response.status_code != 200:
            yield Response(
                content=response.content,
                status_code=response.status_code,
                media_type="application/json",
            )
            return
        yield 'data: {"response":{"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}}\n\n'

    async def fake_wrap(request, _model, project, _enable_credit):
        return {"request": request, "project": project}, "request-id"

    async def fake_retry_config():
        return {"retry_enabled": True, "max_retries": 1, "retry_interval": 0}

    async def no_auto_ban_codes():
        return []

    async def stream_collection_disabled():
        return False

    async def fake_quota_refresh(*_args, **_kwargs):
        return False

    monkeypatch.setattr(antigravity_api, "credential_manager", manager)
    monkeypatch.setattr(antigravity_api, "post_async", fake_post_async)
    monkeypatch.setattr(antigravity_api, "stream_post_async", fake_stream_post_async)
    monkeypatch.setattr(antigravity_api, "wrap_cli_request", fake_wrap)
    monkeypatch.setattr(antigravity_api, "get_retry_config", fake_retry_config)
    monkeypatch.setattr(antigravity_api, "get_auto_ban_error_codes", no_auto_ban_codes)
    monkeypatch.setattr(
        antigravity_api,
        "get_antigravity_stream2nostream",
        stream_collection_disabled,
    )
    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "maybe_schedule",
        fake_quota_refresh,
    )

    request = {"model": "gemini-3.7-flash-low", "request": {}}
    if streaming:
        chunks = [chunk async for chunk in antigravity_api.stream_request(request)]
        assert chunks
        assert all(not isinstance(chunk, Response) for chunk in chunks)
    else:
        response = await antigravity_api.non_stream_request(request)
        assert response.status_code == 200
    assert manager.selection_count == 2
    assert _is_retryable_status(first_status, []) is True
