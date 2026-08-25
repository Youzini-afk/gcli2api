"""Configuration and filesystem paths for the embedded Mihomo service."""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.storage_adapter import get_storage_adapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class MihomoSettings:
    enabled: bool = False
    auto_start: bool = True
    set_as_project_proxy: bool = True
    mixed_port: int = 7897
    controller_host: str = "127.0.0.1"
    controller_port: int = 9097
    secret: str = ""
    profile: str = "default"
    health_check_url: str = "https://www.gstatic.com/generate_204"
    startup_timeout_ms: int = 8000
    download_timeout_ms: int = 15000
    max_subscription_size_bytes: int = 2 * 1024 * 1024
    bin_path: str = ""

    @classmethod
    def from_mapping(cls, value: Any) -> "MihomoSettings":
        source = value if isinstance(value, dict) else {}
        defaults = cls()

        def boolean(key: str, default: bool) -> bool:
            raw = source.get(key, default)
            return raw if isinstance(raw, bool) else default

        def integer(key: str, default: int, minimum: int, maximum: int) -> int:
            raw = source.get(key, default)
            if isinstance(raw, bool):
                return default
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                return default
            return parsed if minimum <= parsed <= maximum else default

        def string(key: str, default: str) -> str:
            raw = source.get(key, default)
            return raw.strip() if isinstance(raw, str) else default

        return cls(
            enabled=boolean("enabled", defaults.enabled),
            auto_start=boolean("auto_start", defaults.auto_start),
            set_as_project_proxy=boolean("set_as_project_proxy", defaults.set_as_project_proxy),
            mixed_port=integer("mixed_port", defaults.mixed_port, 1, 65535),
            controller_host=string("controller_host", defaults.controller_host)
            or defaults.controller_host,
            controller_port=integer("controller_port", defaults.controller_port, 1, 65535),
            secret=string("secret", defaults.secret),
            profile=string("profile", defaults.profile) or defaults.profile,
            health_check_url=string("health_check_url", defaults.health_check_url)
            or defaults.health_check_url,
            startup_timeout_ms=integer(
                "startup_timeout_ms", defaults.startup_timeout_ms, 1000, 60000
            ),
            download_timeout_ms=integer(
                "download_timeout_ms", defaults.download_timeout_ms, 1000, 60000
            ),
            max_subscription_size_bytes=integer(
                "max_subscription_size_bytes",
                defaults.max_subscription_size_bytes,
                64 * 1024,
                20 * 1024 * 1024,
            ),
            bin_path=string("bin_path", defaults.bin_path),
        )

    def to_dict(self, *, include_secret: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_secret:
            result.pop("secret", None)
            result["secret_configured"] = bool(self.secret)
        return result


@dataclass(frozen=True, slots=True)
class MihomoPaths:
    root: Path
    profiles: Path
    providers: Path
    runtime: Path
    state: Path
    runtime_config: Path


def get_paths() -> MihomoPaths:
    root = Path(os.getenv("MIHOMO_DATA_DIR", PROJECT_ROOT / "data" / "mihomo"))
    profiles = root / "profiles"
    providers = root / "providers"
    runtime = root / "runtime"
    for directory in (root, profiles, providers, runtime):
        directory.mkdir(parents=True, exist_ok=True)
    return MihomoPaths(
        root=root,
        profiles=profiles,
        providers=providers,
        runtime=runtime,
        state=root / "state.json",
        runtime_config=runtime / "mihomo.yaml",
    )


def default_binary_name() -> str | None:
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}.get(platform.system())
    machine = platform.machine().lower()
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    if not system or not arch:
        return None
    suffix = ".exe" if system == "windows" else ""
    return f"mihomo-{system}-{arch}{suffix}"


def resolve_binary(settings: MihomoSettings) -> Path | None:
    configured = settings.bin_path or os.getenv("MIHOMO_BIN_PATH", "")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())

    path_binary = shutil.which("mihomo")
    if path_binary:
        candidates.append(Path(path_binary))

    binary_name = default_binary_name()
    if binary_name:
        candidates.extend(
            [PROJECT_ROOT / "bin" / binary_name, get_paths().root / "bin" / binary_name]
        )

    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


async def load_settings() -> MihomoSettings:
    storage = await get_storage_adapter()
    return MihomoSettings.from_mapping(await storage.get_config("mihomo", {}))


async def save_settings(patch: dict[str, Any]) -> MihomoSettings:
    storage = await get_storage_adapter()
    current = await storage.get_config("mihomo", {})
    merged = {**(current if isinstance(current, dict) else {}), **patch}
    normalized = MihomoSettings.from_mapping(merged)
    await storage.set_config("mihomo", normalized.to_dict())
    return normalized
