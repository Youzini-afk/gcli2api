"""Mihomo profile validation, import and runtime configuration generation."""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from .settings import MihomoSettings, get_paths
from .state import patch_state, read_state

NODE_URI_PATTERN = re.compile(
    r"^(ss|ssr|vmess|vless|trojan|hysteria|hysteria2|hy2|tuic|wireguard)://",
    re.IGNORECASE,
)


def normalize_profile_name(name: str | None) -> str:
    value = (name or "default").strip().replace("/", "-").replace("\\", "-")
    value = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", value, flags=re.UNICODE)
    return value[:80] or "default"


def profile_path(name: str) -> Path:
    return get_paths().profiles / f"{normalize_profile_name(name)}.yaml"


def parse_profile(content: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(content or "")
    except yaml.YAMLError as exc:
        raise ValueError(f"Mihomo YAML 无法解析: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Mihomo 配置必须是 YAML 对象")
    return parsed


def build_runtime_config(profile_content: str, settings: MihomoSettings, secret: str) -> str:
    data = parse_profile(profile_content)
    for key in (
        "port",
        "socks-port",
        "redir-port",
        "tproxy-port",
        "mixed-port",
        "allow-lan",
        "bind-address",
        "external-controller",
        "external-controller-tls",
        "external-controller-unix",
        "external-controller-pipe",
        "secret",
        "tun",
        "listeners",
    ):
        data.pop(key, None)

    data["mixed-port"] = settings.mixed_port
    data["allow-lan"] = False
    data["external-controller"] = f"{settings.controller_host}:{settings.controller_port}"
    data["secret"] = secret
    data.setdefault("mode", "rule")
    if not isinstance(data.get("rules"), list) or not data["rules"]:
        data["rules"] = ["MATCH,GLOBAL"]
    if not isinstance(data.get("proxies"), list) and not data.get("proxy-providers"):
        raise ValueError("配置中缺少 proxies 或 proxy-providers")
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=4096)


def _extract_node_uris(content: str) -> list[str]:
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and NODE_URI_PATTERN.match(line.strip())
    ]


def _decode_subscription(content: str) -> str | None:
    compact = re.sub(r"\s+", "", content)
    if len(compact) <= 16 or len(compact) % 4 == 1:
        return None
    try:
        padded = compact.replace("-", "+").replace("_", "/")
        padded += "=" * (-len(padded) % 4)
        decoded = base64.b64decode(padded, validate=False).decode("utf-8").strip()
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if _extract_node_uris(decoded) else None


def _provider_profile(provider_relative_path: str) -> str:
    return yaml.safe_dump(
        {
            "proxy-providers": {
                "subscription": {
                    "type": "file",
                    "path": provider_relative_path,
                    "health-check": {
                        "enable": True,
                        "interval": 600,
                        "url": "https://www.gstatic.com/generate_204",
                    },
                }
            },
            "proxy-groups": [
                {
                    "name": "PROXY",
                    "type": "select",
                    "use": ["subscription"],
                    "proxies": ["DIRECT"],
                },
                {
                    "name": "Auto",
                    "type": "url-test",
                    "use": ["subscription"],
                    "url": "https://www.gstatic.com/generate_204",
                    "interval": 300,
                },
            ],
            "rules": ["MATCH,PROXY"],
        },
        allow_unicode=True,
        sort_keys=False,
        width=4096,
    )


def _is_non_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return not ip.is_global


async def assert_public_subscription_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("订阅 URL 仅支持有效的 http/https 地址")
    if parsed.username or parsed.password:
        raise ValueError("订阅 URL 不支持内嵌用户名或密码")
    if parsed.hostname.lower() == "metadata.google.internal":
        raise ValueError("订阅 URL 不允许访问云元数据地址")

    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError("订阅 URL 域名无法解析") from exc
    addresses = {record[4][0] for record in records}
    if not addresses or any(_is_non_public(address) for address in addresses):
        raise ValueError("订阅 URL 解析到内网、本机或保留地址，已拒绝")
    return parsed.geturl()


async def _download_profile(url: str, settings: MihomoSettings) -> str:
    timeout = settings.download_timeout_ms / 1000
    total = 0
    chunks: list[bytes] = []
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=False, trust_env=False
    ) as client:
        async with client.stream(
            "GET", url, headers={"User-Agent": "gcli2api-mihomo/1.0"}
        ) as response:
            if response.is_redirect:
                raise ValueError("订阅地址发生重定向，请直接填写最终 URL")
            if response.is_error:
                raise ValueError(f"订阅下载失败: HTTP {response.status_code}")
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > settings.max_subscription_size_bytes:
                    raise ValueError(
                        f"订阅内容超过配置上限 {settings.max_subscription_size_bytes} bytes"
                    )
                chunks.append(chunk)
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("订阅内容不是 UTF-8 文本") from exc


async def _upsert_profile(meta: dict[str, Any]) -> dict[str, Any]:
    state = await read_state()
    profiles = [item for item in state["profiles"] if item.get("name") != meta["name"]]
    profiles.insert(0, meta)
    await patch_state({"profiles": profiles, "current_profile": meta["name"], "last_error": ""})
    return meta


async def import_profile_from_yaml(name: str, content: str) -> dict[str, Any]:
    safe_name = normalize_profile_name(name or "local")
    raw = (content or "").strip()
    if not raw:
        raise ValueError("YAML 内容为空")
    parse_profile(raw)
    # Runtime validation also ensures a usable proxy/provider section exists.
    build_runtime_config(raw, MihomoSettings(), "validation")
    path = profile_path(safe_name)
    path.write_text(raw, encoding="utf-8")
    return await _upsert_profile(
        {
            "name": safe_name,
            "source": "local",
            "url": "",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )


async def import_profile_from_url(name: str, url: str, settings: MihomoSettings) -> dict[str, Any]:
    safe_url = await assert_public_subscription_url(url)
    safe_name = normalize_profile_name(name or urlparse(safe_url).hostname or "subscription")
    content = (await _download_profile(safe_url, settings)).strip()
    source = "url"
    try:
        build_runtime_config(content, MihomoSettings(), "validation")
        profile_content = content
    except ValueError as yaml_error:
        decoded = _decode_subscription(content)
        node_content = decoded or content
        if not _extract_node_uris(node_content):
            raise yaml_error
        provider = get_paths().providers / f"{safe_name}.sub"
        provider.write_text(node_content, encoding="utf-8")
        profile_content = _provider_profile(f"providers/{provider.name}")
        source = "url:file-provider"

    profile_path(safe_name).write_text(profile_content, encoding="utf-8")
    return await _upsert_profile(
        {
            "name": safe_name,
            "source": source,
            "url": safe_url,
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )


async def list_profiles() -> dict[str, Any]:
    state = await read_state()
    return {
        "current_profile": state["current_profile"],
        "profiles": state["profiles"],
    }


async def remove_profile(name: str) -> dict[str, Any]:
    safe_name = normalize_profile_name(name)
    state = await read_state()
    profile_path(safe_name).unlink(missing_ok=True)
    (get_paths().providers / f"{safe_name}.sub").unlink(missing_ok=True)
    profiles = [item for item in state["profiles"] if item.get("name") != safe_name]
    current = state["current_profile"]
    if current == safe_name:
        current = profiles[0]["name"] if profiles else "default"
    return await patch_state({"profiles": profiles, "current_profile": current})


def write_runtime_config(profile_name: str, settings: MihomoSettings, secret: str) -> Path:
    source = profile_path(profile_name)
    if not source.is_file():
        raise FileNotFoundError(f"Mihomo Profile 不存在: {profile_name}")
    runtime = get_paths().runtime_config
    runtime.write_text(
        build_runtime_config(source.read_text(encoding="utf-8"), settings, secret),
        encoding="utf-8",
    )
    return runtime
