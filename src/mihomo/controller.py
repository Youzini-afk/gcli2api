"""Async client for the local Mihomo external controller API."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

from .settings import MihomoSettings


class MihomoControllerError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class MihomoController:
    def __init__(self, settings: MihomoSettings, secret: str):
        self.base_url = f"http://{settings.controller_host}:{settings.controller_port}"
        self.secret = secret

    async def request(self, method: str, path: str, *, timeout: float = 5.0, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if self.secret:
            headers["Authorization"] = f"Bearer {self.secret}"
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=timeout, trust_env=False
            ) as client:
                response = await client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as exc:
            raise MihomoControllerError(f"无法连接 Mihomo 控制器: {exc}", 503) from exc

        if response.status_code == 204:
            return None
        try:
            data = response.json()
        except ValueError:
            data = response.text
        if response.is_error:
            if isinstance(data, dict):
                message = data.get("message") or data.get("error") or str(data)
            else:
                message = str(data) or f"HTTP {response.status_code}"
            raise MihomoControllerError(message, response.status_code)
        return data

    async def version(self) -> Any:
        return await self.request("GET", "/version")

    async def proxies(self) -> Any:
        return await self.request("GET", "/proxies")

    async def select_proxy(self, group: str, name: str) -> Any:
        return await self.request("PUT", f"/proxies/{quote(group, safe='')}", json={"name": name})

    async def test_delay(self, name: str, url: str, timeout_ms: int) -> Any:
        return await self.request(
            "GET",
            f"/proxies/{quote(name, safe='')}/delay",
            timeout=(timeout_ms / 1000) + 2,
            params={"url": url, "timeout": timeout_ms},
        )
