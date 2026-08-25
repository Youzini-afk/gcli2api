"""Small local runtime state store for Mihomo profiles and selections."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from .settings import get_paths

DEFAULT_STATE: dict[str, Any] = {
    "profiles": [],
    "current_profile": "default",
    "selected": {},
    "last_started_at": None,
    "last_error": "",
}

_state_lock = asyncio.Lock()


def _read_sync() -> dict[str, Any]:
    path = get_paths().state
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("state must be an object")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        parsed = {}
    return {
        **DEFAULT_STATE,
        **parsed,
        "profiles": (
            parsed.get("profiles", []) if isinstance(parsed.get("profiles", []), list) else []
        ),
        "selected": (
            parsed.get("selected", {}) if isinstance(parsed.get("selected", {}), dict) else {}
        ),
    }


def _write_sync(state: dict[str, Any]) -> dict[str, Any]:
    path = get_paths().state
    normalized = {
        **DEFAULT_STATE,
        **state,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return normalized


async def read_state() -> dict[str, Any]:
    async with _state_lock:
        return await asyncio.to_thread(_read_sync)


async def patch_state(patch: dict[str, Any]) -> dict[str, Any]:
    async with _state_lock:
        current = await asyncio.to_thread(_read_sync)
        return await asyncio.to_thread(_write_sync, {**current, **patch})
