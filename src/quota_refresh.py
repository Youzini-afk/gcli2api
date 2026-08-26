"""Request-driven Antigravity quota snapshots without periodic polling."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any

from log import log
from src.google_oauth_api import Credentials
from src.storage_adapter import get_storage_adapter

DEFAULT_QUOTA_STALE_SECONDS = 5 * 60
QUOTA_COOLDOWN_THRESHOLD = 0.05


def summarize_quota_pool(
    models: dict[str, Any], model_name: str
) -> tuple[float | None, str | None]:
    """Return the minimum remaining quota and earliest reset for a shared pool."""
    wants_claude = "claude" in model_name.lower()
    remaining_values: list[float] = []
    reset_timestamps: list[tuple[float, str]] = []
    for name, quota in models.items():
        if ("claude" in name.lower()) != wants_claude or not isinstance(quota, dict):
            continue
        try:
            remaining_values.append(max(0.0, min(1.0, float(quota.get("remaining", 0)))))
        except (TypeError, ValueError):
            continue
        reset_raw = quota.get("resetTimeRaw")
        if reset_raw:
            try:
                parsed = datetime.fromisoformat(str(reset_raw).replace("Z", "+00:00")).timestamp()
                reset_timestamps.append((parsed, str(reset_raw)))
            except (TypeError, ValueError, OverflowError):
                pass
    remaining = min(remaining_values) if remaining_values else None
    reset_at = min(reset_timestamps)[1] if reset_timestamps else None
    return remaining, reset_at


async def reconcile_healthy_pool_cooldowns(
    storage: Any, filename: str, models: dict[str, Any]
) -> list[str]:
    """Clear stale pool cooldowns when refreshed quota proves the pool is healthy."""
    from src.credential_manager import (
        ANTIGRAVITY_CLAUDE_POOL_KEY,
        ANTIGRAVITY_GEMINI_POOL_KEY,
    )

    state = await storage.get_credential_state(filename, mode="antigravity")
    cooldowns = state.get("model_cooldowns", {})
    if not isinstance(cooldowns, dict) or not cooldowns:
        return []
    cleared: list[str] = []
    for representative, pool_key in (
        ("gemini", ANTIGRAVITY_GEMINI_POOL_KEY),
        ("claude", ANTIGRAVITY_CLAUDE_POOL_KEY),
    ):
        remaining, _ = summarize_quota_pool(models, representative)
        if pool_key in cooldowns and remaining is not None and remaining > QUOTA_COOLDOWN_THRESHOLD:
            await storage._backend.set_model_cooldown(filename, pool_key, None, mode="antigravity")
            cleared.append(pool_key)
    if cleared:
        log.info(f"额度已恢复，清除 Antigravity 旧冷却: {filename}, pools={cleared}")
    return cleared


class QuotaRefreshService:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._last_refresh: dict[str, float] = {}
        self._last_attempt: dict[str, float] = {}
        self._last_event: dict[str, str] = {}
        self._lock = asyncio.Lock()

    @property
    def stale_after_seconds(self) -> float:
        raw = os.getenv("QUOTA_REFRESH_STALE_SECONDS", "")
        if not raw:
            return DEFAULT_QUOTA_STALE_SECONDS
        try:
            return max(0.0, float(raw))
        except ValueError:
            return DEFAULT_QUOTA_STALE_SECONDS

    async def _load_refresh_times(self, filename: str) -> tuple[float, float]:
        if filename in self._last_refresh and filename in self._last_attempt:
            return self._last_refresh[filename], self._last_attempt[filename]
        storage = await get_storage_adapter()
        state = await storage.get_credential_state(filename, mode="antigravity")
        try:
            last_refresh = float(state.get("quota_updated_at", 0))
        except (TypeError, ValueError):
            last_refresh = 0.0
        try:
            last_attempt = float(state.get("quota_checked_at", last_refresh))
        except (TypeError, ValueError):
            last_attempt = last_refresh
        self._last_refresh[filename] = last_refresh
        self._last_attempt[filename] = last_attempt
        return last_refresh, last_attempt

    async def get_cached(self, filename: str) -> dict[str, Any] | None:
        storage = await get_storage_adapter()
        state = await storage.get_credential_state(filename, mode="antigravity")
        models = state.get("quota_models")
        if not isinstance(models, dict):
            return None
        try:
            updated_at = float(state.get("quota_updated_at", 0))
        except (TypeError, ValueError):
            return None
        if time.time() - updated_at >= self.stale_after_seconds:
            return None
        return {
            "success": True,
            "filename": filename,
            "models": models,
            "updated_at": updated_at,
            "stale_after_seconds": self.stale_after_seconds,
            "cached": True,
        }

    async def maybe_schedule(
        self,
        filename: str,
        *,
        success: bool,
        status_code: int | None = None,
    ) -> bool:
        """Refresh on the first request after staleness or on transition into HTTP 429."""
        if not filename:
            return False
        event = "success" if success else f"error:{status_code or 0}"
        previous_event = self._last_event.get(filename)
        self._last_event[filename] = event
        entered_429 = event == "error:429" and previous_event != "error:429"
        last_refresh, last_attempt = await self._load_refresh_times(filename)
        stale = time.time() - max(last_refresh, last_attempt) >= self.stale_after_seconds
        if not stale and not entered_429:
            return False
        async with self._lock:
            existing = self._tasks.get(filename)
            if existing and not existing.done():
                return False
            self._start_task(filename)
        return True

    async def refresh_now(self, filename: str) -> dict[str, Any]:
        async with self._lock:
            task = self._tasks.get(filename)
            if not task or task.done():
                task = self._start_task(filename)
        return await asyncio.shield(task)

    async def inspect_pool_after_429(self, filename: str, model_name: str) -> dict[str, Any]:
        """Refresh quota after a 429 and decide whether it is actual pool exhaustion."""
        # This synchronous check already refreshes quota; prevent the later
        # error recorder from scheduling the same 429 transition a second time.
        self._last_event[filename] = "error:429"
        try:
            quota = await self.refresh_now(filename)
        except Exception as error:
            return {
                "confirmed_exhausted": False,
                "remaining": None,
                "reset_at": None,
                "error": str(error),
            }
        remaining, reset_at = summarize_quota_pool(quota.get("models", {}), model_name)
        return {
            "confirmed_exhausted": (
                remaining is not None and remaining <= QUOTA_COOLDOWN_THRESHOLD
            ),
            "remaining": remaining,
            "reset_at": reset_at,
            "error": None,
        }

    async def recover_one_healthy_cooldown(
        self,
        model_name: str,
        exclude_filenames: set[str] | None = None,
    ) -> bool:
        """Recheck cooled credentials on demand and recover the first healthy one."""
        from src.credential_manager import normalize_model_cooldown_key

        storage = await get_storage_adapter()
        pool_key = normalize_model_cooldown_key("antigravity", model_name)
        excluded = exclude_filenames or set()
        filenames = sorted(await storage.list_credentials(mode="antigravity"))
        now = time.time()
        for filename in filenames:
            if filename in excluded:
                continue
            state = await storage.get_credential_state(filename, mode="antigravity")
            if state.get("disabled"):
                continue
            cooldowns = state.get("model_cooldowns", {})
            cooldown_until = cooldowns.get(pool_key) if isinstance(cooldowns, dict) else None
            if not isinstance(cooldown_until, (int, float)) or cooldown_until <= now:
                continue
            try:
                quota = await self.refresh_now(filename)
            except Exception as error:
                log.debug(f"恢复冷却检查失败 {filename}: {error}")
                continue
            remaining, _ = summarize_quota_pool(quota.get("models", {}), model_name)
            if remaining is not None and remaining > QUOTA_COOLDOWN_THRESHOLD:
                log.info(
                    f"按请求恢复 Antigravity 凭证: {filename}, "
                    f"pool={pool_key}, remaining={remaining * 100:.1f}%"
                )
                return True
        return False

    def _start_task(self, filename: str) -> asyncio.Task:
        self._last_attempt[filename] = time.time()
        task = asyncio.create_task(self._refresh(filename), name=f"quota-refresh:{filename}")
        self._tasks[filename] = task
        task.add_done_callback(lambda completed, key=filename: self._task_done(key, completed))
        return task

    def _task_done(self, filename: str, task: asyncio.Task) -> None:
        if self._tasks.get(filename) is task:
            self._tasks.pop(filename, None)
        if task.cancelled():
            return
        error = task.exception()
        if error:
            log.warning(f"后台刷新 Antigravity 额度失败 {filename}: {error}")

    async def _refresh(self, filename: str) -> dict[str, Any]:
        storage = await get_storage_adapter()
        credential_data = await storage.get_credential(filename, mode="antigravity")
        if not credential_data:
            raise ValueError("凭证不存在")

        credentials = Credentials.from_dict(credential_data)
        token_refreshed = await credentials.refresh_if_needed()
        if token_refreshed:
            credential_data = credentials.to_dict()
            await storage.store_credential(filename, credential_data, mode="antigravity")

        access_token = credential_data.get("access_token") or credential_data.get("token")
        if not access_token:
            raise ValueError("凭证中没有访问令牌")

        # Runtime import avoids a module cycle with src.api.utils.
        from src.api.antigravity import fetch_quota_info

        quota_info = await fetch_quota_info(access_token)
        checked_at = time.time()
        self._last_attempt[filename] = checked_at
        if not quota_info.get("success"):
            error = quota_info.get("error", "未知错误")
            await storage.update_credential_state(
                filename,
                {"quota_checked_at": checked_at, "quota_last_error": error},
                mode="antigravity",
            )
            raise RuntimeError(error)

        updated_at = checked_at
        models = quota_info.get("models", {})
        await reconcile_healthy_pool_cooldowns(storage, filename, models)
        await storage.update_credential_state(
            filename,
            {
                "quota_models": models,
                "quota_updated_at": updated_at,
                "quota_checked_at": checked_at,
                "quota_last_error": "",
            },
            mode="antigravity",
        )
        self._last_refresh[filename] = updated_at
        log.debug(f"已按请求刷新 Antigravity 额度: {filename}")
        return {
            "success": True,
            "filename": filename,
            "models": models,
            "updated_at": updated_at,
            "stale_after_seconds": self.stale_after_seconds,
            "cached": False,
        }

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


quota_refresh_service = QuotaRefreshService()
