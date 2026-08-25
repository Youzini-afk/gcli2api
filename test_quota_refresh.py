import time

import pytest

import src.quota_refresh as quota_refresh_module
from src.api.utils import record_api_call_error, record_api_call_success
from src.quota_refresh import QuotaRefreshService


@pytest.fixture(autouse=True)
def fixed_quota_refresh_window(monkeypatch):
    monkeypatch.setenv("QUOTA_REFRESH_STALE_SECONDS", "300")


@pytest.mark.asyncio
async def test_stale_request_schedules_one_background_refresh(monkeypatch):
    service = QuotaRefreshService()
    service._last_refresh["account.json"] = 0
    service._last_attempt["account.json"] = 0
    calls = []

    async def fake_refresh(filename):
        calls.append(filename)
        return {"success": True, "filename": filename, "models": {}}

    monkeypatch.setattr(service, "_refresh", fake_refresh)

    assert await service.maybe_schedule("account.json", success=True) is True
    task = service._tasks["account.json"]
    assert await service.maybe_schedule("account.json", success=True) is False
    await task

    assert calls == ["account.json"]


@pytest.mark.asyncio
async def test_failed_refresh_attempt_is_throttled_until_snapshot_window(monkeypatch):
    service = QuotaRefreshService()
    service._last_refresh["account.json"] = 0
    service._last_attempt["account.json"] = 0

    async def fake_refresh(_filename):
        raise RuntimeError("upstream unavailable")

    monkeypatch.setattr(service, "_refresh", fake_refresh)

    assert await service.maybe_schedule("account.json", success=True) is True
    task = service._tasks["account.json"]
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        await task

    assert await service.maybe_schedule("account.json", success=True) is False


@pytest.mark.asyncio
async def test_first_429_transition_refreshes_even_when_snapshot_is_fresh(monkeypatch):
    service = QuotaRefreshService()
    now = time.time()
    service._last_refresh["account.json"] = now
    service._last_attempt["account.json"] = now
    calls = []

    async def fake_refresh(filename):
        calls.append(filename)
        return {"success": True, "filename": filename, "models": {}}

    monkeypatch.setattr(service, "_refresh", fake_refresh)

    assert await service.maybe_schedule("account.json", success=False, status_code=429) is True
    await service._tasks["account.json"]
    assert await service.maybe_schedule("account.json", success=False, status_code=429) is False

    assert await service.maybe_schedule("account.json", success=True) is False
    assert await service.maybe_schedule("account.json", success=False, status_code=429) is True
    await service._tasks["account.json"]

    assert calls == ["account.json", "account.json"]


@pytest.mark.asyncio
async def test_cached_snapshot_is_only_returned_while_fresh(monkeypatch):
    state = {
        "quota_models": {"gemini-test": {"remaining": 0.5}},
        "quota_updated_at": time.time(),
    }

    class FakeStorage:
        async def get_credential_state(self, filename, mode):
            assert filename == "account.json"
            assert mode == "antigravity"
            return state

    async def fake_storage_adapter():
        return FakeStorage()

    monkeypatch.setattr(quota_refresh_module, "get_storage_adapter", fake_storage_adapter)
    service = QuotaRefreshService()

    cached = await service.get_cached("account.json")
    assert cached is not None
    assert cached["cached"] is True

    state["quota_updated_at"] = time.time() - service.stale_after_seconds - 1
    assert await service.get_cached("account.json") is None


@pytest.mark.asyncio
async def test_antigravity_request_results_drive_quota_refresh_checks(monkeypatch):
    recorded = []
    refresh_events = []

    class FakeCredentialManager:
        async def record_api_call_result(self, *args, **kwargs):
            recorded.append((args, kwargs))

    async def fake_maybe_schedule(filename, *, success, status_code=None):
        refresh_events.append((filename, success, status_code))
        return True

    monkeypatch.setattr(
        quota_refresh_module.quota_refresh_service,
        "maybe_schedule",
        fake_maybe_schedule,
    )
    manager = FakeCredentialManager()

    await record_api_call_success(manager, "account.json", mode="antigravity")
    await record_api_call_error(manager, "account.json", 429, mode="antigravity")

    assert len(recorded) == 2
    assert refresh_events == [
        ("account.json", True, None),
        ("account.json", False, 429),
    ]
