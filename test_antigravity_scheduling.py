from datetime import datetime

import pytest
from fastapi import Response

import src.api.antigravity as antigravity_api
import src.api.utils as api_utils
import src.quota_refresh as quota_refresh_module
from src.api.antigravity import get_confirmed_429_cooldown, _is_retryable_status
from src.credential_manager import (
    ANTIGRAVITY_CLAUDE_POOL_KEY,
    ANTIGRAVITY_GEMINI_POOL_KEY,
    CredentialManager,
    normalize_model_cooldown_key,
)
from src.quota_refresh import reconcile_healthy_pool_cooldowns, summarize_quota_pool
from src.storage.sqlite_manager import SQLiteManager


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


def test_message_only_quota_reset_delay_supports_spaces_and_fractional_seconds(monkeypatch):
    monkeypatch.setattr(api_utils.time, "time", lambda: 1_000.0)
    error = {"error": {"message": "Your quota will reset after 1h 2m 3.5s."}}
    assert api_utils.parse_quota_reset_timestamp(error, mode="geminicli") == 4723.5


def test_antigravity_generic_429_does_not_invent_a_four_hour_reset():
    error = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "message": "Resource has been exhausted (e.g. check quota).",
        }
    }
    assert api_utils.parse_quota_reset_timestamp(error, mode="antigravity") is None


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


def test_quota_confirmation_uses_the_requested_shared_pool():
    models = {
        "gemini-3.7-flash-low": {
            "remaining": 0.84,
            "resetTimeRaw": "2030-09-01T15:40:16Z",
        },
        "gemini-3-pro-high": {
            "remaining": 0.91,
            "resetTimeRaw": "2030-09-01T16:40:16Z",
        },
        "claude-sonnet-4-5": {
            "remaining": 0.03,
            "resetTimeRaw": "2030-08-31T15:40:16Z",
        },
    }

    gemini_remaining, _ = summarize_quota_pool(models, "gemini-3.7-flash-low")
    claude_remaining, _ = summarize_quota_pool(models, "claude-sonnet-4-5")

    assert gemini_remaining == 0.84
    assert claude_remaining == 0.03


@pytest.mark.asyncio
async def test_credential_manager_selects_and_records_by_antigravity_pool(
    monkeypatch,
):
    calls = []

    class FakeBackend:
        async def get_next_available_credential(self, *, mode, model_name, exclude_filenames):
            calls.append(("select", mode, model_name, exclude_filenames))
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

    async def no_recovery(*_args, **_kwargs):
        return False

    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "recover_one_healthy_cooldown",
        no_recovery,
    )

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

    assert ("select", "antigravity", ANTIGRAVITY_GEMINI_POOL_KEY, None) in calls
    assert (
        "cooldown",
        "account.json",
        ANTIGRAVITY_GEMINI_POOL_KEY,
        2_000_000_000.0,
        "antigravity",
    ) in calls


@pytest.mark.asyncio
async def test_healthy_quota_clears_stale_pool_cooldown():
    cleared = []

    class FakeBackend:
        async def set_model_cooldown(self, filename, model_name, cooldown_until, *, mode):
            cleared.append((filename, model_name, cooldown_until, mode))

    class FakeStorage:
        _backend = FakeBackend()

        async def get_credential_state(self, _filename, *, mode):
            assert mode == "antigravity"
            return {
                "model_cooldowns": {
                    ANTIGRAVITY_GEMINI_POOL_KEY: 2_000_000_000.0,
                }
            }

    await reconcile_healthy_pool_cooldowns(
        FakeStorage(),
        "account.json",
        {"gemini-3.7-flash-low": {"remaining": 0.82}},
    )

    assert cleared == [
        (
            "account.json",
            ANTIGRAVITY_GEMINI_POOL_KEY,
            None,
            "antigravity",
        )
    ]


@pytest.mark.asyncio
async def test_no_available_pool_rechecks_cooldown_then_selects_recovered_account(
    monkeypatch,
):
    selections = 0

    class FakeBackend:
        async def get_next_available_credential(self, **_kwargs):
            nonlocal selections
            selections += 1
            if selections == 1:
                return None
            return "recovered.json", {
                "access_token": "token",
                "project_id": "project",
                "expiry": "2030-01-01T00:00:00+00:00",
            }

    class FakeStorage:
        _backend = FakeBackend()

    async def recover(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "recover_one_healthy_cooldown",
        recover,
    )
    manager = CredentialManager()
    manager._initialized = True
    manager._storage_adapter = FakeStorage()

    selected = await manager.get_valid_credential(
        mode="antigravity", model_name="gemini-3.7-flash-low"
    )

    assert selected is not None
    assert selected[0] == "recovered.json"
    assert selections == 2


@pytest.mark.asyncio
async def test_sqlite_retry_excludes_credentials_already_attempted(monkeypatch, tmp_path):
    monkeypatch.setenv("CREDENTIALS_DIR", str(tmp_path))
    storage = SQLiteManager()
    await storage.initialize()
    try:
        await storage.store_credential("first.json", {"access_token": "first"}, mode="antigravity")
        await storage.store_credential(
            "second.json", {"access_token": "second"}, mode="antigravity"
        )

        selected = await storage.get_next_available_credential(
            mode="antigravity",
            model_name=ANTIGRAVITY_GEMINI_POOL_KEY,
            exclude_filenames={"first.json"},
        )

        assert selected is not None
        assert selected[0] == "second.json"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status", [400, 403, 429])
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

        async def get_valid_credential(self, *, mode, model_name, exclude_filenames=None):
            self.selection_count += 1
            if self.selection_count == 1:
                assert not exclude_filenames
                return "bad.json", {
                    "access_token": "bad",
                    "project_id": "bad-project",
                }
            assert self.error_recorded is True
            assert "bad.json" in exclude_filenames
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

    async def fake_pool_inspection(*_args, **_kwargs):
        return {
            "confirmed_exhausted": True,
            "remaining": 0.04,
            "reset_at": None,
            "error": None,
        }

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
    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "inspect_pool_after_429",
        fake_pool_inspection,
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


@pytest.mark.asyncio
async def test_transient_429_with_healthy_quota_does_not_persist_cooldown(
    monkeypatch,
):
    async def healthy_pool(*_args, **_kwargs):
        return {
            "confirmed_exhausted": False,
            "remaining": 0.84,
            "reset_at": "2030-09-01T15:40:16Z",
            "error": None,
        }

    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "inspect_pool_after_429",
        healthy_pool,
    )

    cooldown = await get_confirmed_429_cooldown(
        "account.json",
        "gemini-3.7-flash-low",
        '{"error":{"status":"RESOURCE_EXHAUSTED","details":' '[{"retryDelay":"155h"}]}}',
    )

    assert cooldown is None
