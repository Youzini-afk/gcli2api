"""Lifecycle manager for the bundled or user-provided Mihomo executable."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections import deque
from datetime import UTC, datetime
from typing import Any

import config
from log import log

from .controller import MihomoController, MihomoControllerError
from .profiles import write_runtime_config
from .settings import MihomoSettings, get_paths, load_settings, resolve_binary
from .state import patch_state, read_state


class MihomoManager:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.started_at: str | None = None
        self.logs: deque[dict[str, str]] = deque(maxlen=200)
        self._generated_secret = ""
        self._lifecycle_lock = asyncio.Lock()
        self._reader_tasks: list[asyncio.Task] = []
        self._watcher_task: asyncio.Task | None = None
        self._stopping = False

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _secret(self, settings: MihomoSettings) -> str:
        if settings.secret:
            return settings.secret
        if not self._generated_secret:
            self._generated_secret = secrets.token_hex(24)
        return self._generated_secret

    async def controller(self) -> MihomoController:
        settings = await load_settings()
        return MihomoController(settings, self._secret(settings))

    async def _read_stream(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                entry = {"time": datetime.now(UTC).isoformat(), "line": text}
                self.logs.append(entry)
                log.debug(f"[mihomo] {text}")

    def _apply_project_proxy(self, settings: MihomoSettings) -> None:
        if settings.enabled and settings.set_as_project_proxy and self.is_running():
            config.set_runtime_proxy(f"http://127.0.0.1:{settings.mixed_port}")
        else:
            config.set_runtime_proxy(None)

    async def _watch_process(
        self, process: asyncio.subprocess.Process, settings: MihomoSettings
    ) -> None:
        return_code = await process.wait()
        if self.process is process:
            self.process = None
            self._apply_project_proxy(settings)
        if not self._stopping:
            message = f"Mihomo 已退出: code={return_code}"
            log.warning(message)
            await patch_state({"last_error": message})

    async def _wait_ready(self, settings: MihomoSettings) -> None:
        deadline = asyncio.get_running_loop().time() + settings.startup_timeout_ms / 1000
        controller = MihomoController(settings, self._secret(settings))
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if not self.is_running():
                break
            try:
                await controller.version()
                return
            except Exception as exc:  # Controller connection errors are expected during startup.
                last_error = exc
                await asyncio.sleep(0.3)
        message = str(last_error) if last_error else "Mihomo 进程提前退出"
        raise RuntimeError(f"Mihomo 控制端口未就绪: {message}")

    async def start(self, profile: str | None = None) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self.is_running():
                return await self.status()
            settings = await load_settings()
            selected_profile = (profile or settings.profile).strip()
            try:
                if int(os.getenv("WORKERS", "1")) > 1:
                    raise ValueError("Mihomo 进程管理要求 WORKERS=1；多 worker 会争抢同一控制端口")
                binary = resolve_binary(settings)
                if binary is None:
                    raise FileNotFoundError(
                        "未找到 Mihomo 核心；请配置 bin_path、MIHOMO_BIN_PATH，或将 mihomo 加入 PATH"
                    )
                if os.name != "nt":
                    binary.chmod(binary.stat().st_mode | 0o111)

                runtime = write_runtime_config(selected_profile, settings, self._secret(settings))
            except Exception as exc:
                await patch_state({"last_error": str(exc)})
                raise
            paths = get_paths()
            self._stopping = False
            self.process = await asyncio.create_subprocess_exec(
                str(binary),
                "-d",
                str(paths.root),
                "-f",
                str(runtime),
                "-secret",
                self._secret(settings),
                "-ext-ctl",
                f"{settings.controller_host}:{settings.controller_port}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self.started_at = datetime.now(UTC).isoformat()
            self._reader_tasks = [
                asyncio.create_task(self._read_stream(self.process.stdout)),
                asyncio.create_task(self._read_stream(self.process.stderr)),
            ]
            self._watcher_task = asyncio.create_task(self._watch_process(self.process, settings))
            try:
                await self._wait_ready(settings)
            except Exception as exc:
                await self._stop_locked(settings)
                await patch_state({"last_error": str(exc)})
                raise

            await patch_state(
                {
                    "current_profile": selected_profile,
                    "last_started_at": self.started_at,
                    "last_error": "",
                }
            )
            await self._restore_selections(settings)
            self._apply_project_proxy(settings)
            log.info(f"Mihomo 已启动: 127.0.0.1:{settings.mixed_port}")
            return await self.status()

    async def _restore_selections(self, settings: MihomoSettings) -> None:
        selected = (await read_state()).get("selected", {})
        controller = MihomoController(settings, self._secret(settings))
        for group, name in selected.items():
            if not group or not name:
                continue
            try:
                await controller.select_proxy(group, name)
            except MihomoControllerError as exc:
                log.debug(f"恢复 Mihomo 节点选择失败 {group} -> {name}: {exc}")

    async def _stop_locked(self, settings: MihomoSettings | None = None) -> None:
        process = self.process
        self.process = None
        self._stopping = True
        try:
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            for task in self._reader_tasks:
                if not task.done():
                    task.cancel()
            if self._reader_tasks:
                await asyncio.gather(*self._reader_tasks, return_exceptions=True)
            self._reader_tasks = []
            if self._watcher_task and not self._watcher_task.done():
                self._watcher_task.cancel()
                await asyncio.gather(self._watcher_task, return_exceptions=True)
            self._watcher_task = None
        finally:
            self._stopping = False
            self._apply_project_proxy(settings or await load_settings())

    async def stop(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            await self._stop_locked()
            return await self.status()

    async def restart(self, profile: str | None = None) -> dict[str, Any]:
        async with self._lifecycle_lock:
            await self._stop_locked()
            self._generated_secret = ""
        return await self.start(profile)

    async def on_settings_updated(
        self, previous: MihomoSettings, current: MihomoSettings
    ) -> dict[str, Any]:
        if not current.enabled:
            return await self.stop()
        restart_fields = (
            "mixed_port",
            "controller_host",
            "controller_port",
            "secret",
            "profile",
            "bin_path",
            "startup_timeout_ms",
        )
        if self.is_running() and any(
            getattr(previous, key) != getattr(current, key) for key in restart_fields
        ):
            return await self.restart(current.profile)
        if current.auto_start and not self.is_running():
            return await self.start(current.profile)
        self._apply_project_proxy(current)
        return await self.status()

    async def auto_start(self) -> None:
        settings = await load_settings()
        if not settings.enabled or not settings.auto_start:
            return
        try:
            await self.start(settings.profile)
        except Exception as exc:
            log.warning(f"Mihomo 自动启动失败: {exc}")

    async def status(self) -> dict[str, Any]:
        settings = await load_settings()
        state = await read_state()
        binary = resolve_binary(settings)
        return {
            **settings.to_dict(include_secret=False),
            "running": self.is_running(),
            "pid": self.process.pid if self.is_running() and self.process else None,
            "started_at": self.started_at,
            "controller": f"{settings.controller_host}:{settings.controller_port}",
            "project_proxy": (
                f"http://127.0.0.1:{settings.mixed_port}"
                if settings.enabled and settings.set_as_project_proxy
                else await config.get_config_value("proxy", "", "PROXY")
            ),
            "proxy_active": bool(
                self.is_running() and settings.enabled and settings.set_as_project_proxy
            ),
            "binary_path": str(binary) if binary else settings.bin_path,
            "binary_available": binary is not None,
            "current_profile": state["current_profile"],
            "profiles": state["profiles"],
            "selected": state["selected"],
            "last_error": state["last_error"],
            "logs": list(self.logs)[-50:],
        }


mihomo_manager = MihomoManager()
