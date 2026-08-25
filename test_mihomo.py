from pathlib import Path
import importlib

import pytest
import yaml
from fastapi import HTTPException

from src.mihomo.profiles import (
    assert_public_subscription_url,
    build_runtime_config,
    import_profile_from_yaml,
    list_profiles,
    normalize_profile_name,
)
from src.mihomo.settings import MihomoSettings, get_paths
from src.panel import mihomo as panel_mihomo


def test_profile_name_is_safe_and_readable():
    assert normalize_profile_name(" Hong Kong / Premium ") == "Hong_Kong_-_Premium"
    assert normalize_profile_name("../../escape") == "..-..-escape"


def test_runtime_config_owns_local_ports_and_disables_lan():
    source = yaml.safe_dump(
        {
            "mixed-port": 7890,
            "allow-lan": True,
            "external-controller": "0.0.0.0:9090",
            "secret": "from-profile",
            "tun": {"enable": True},
            "proxies": [{"name": "direct", "type": "direct"}],
            "rules": [],
        }
    )
    settings = MihomoSettings(mixed_port=7897, controller_port=9097)

    runtime = yaml.safe_load(build_runtime_config(source, settings, "runtime-secret"))

    assert runtime["mixed-port"] == 7897
    assert runtime["allow-lan"] is False
    assert runtime["external-controller"] == "127.0.0.1:9097"
    assert runtime["secret"] == "runtime-secret"
    assert runtime["rules"] == ["MATCH,GLOBAL"]
    assert "tun" not in runtime


def test_runtime_config_requires_a_proxy_source():
    with pytest.raises(ValueError, match="proxies"):
        build_runtime_config("rules:\n  - MATCH,DIRECT\n", MihomoSettings(), "secret")


@pytest.mark.asyncio
async def test_subscription_rejects_local_and_metadata_addresses():
    for url in (
        "http://127.0.0.1/sub",
        "http://10.0.0.1/sub",
        "http://[::1]/sub",
        "http://metadata.google.internal/sub",
    ):
        with pytest.raises(ValueError):
            await assert_public_subscription_url(url)


@pytest.mark.asyncio
async def test_yaml_profile_import_persists_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MIHOMO_DATA_DIR", str(tmp_path / "mihomo"))
    content = yaml.safe_dump(
        {
            "proxies": [{"name": "direct", "type": "direct"}],
            "proxy-groups": [{"name": "PROXY", "type": "select", "proxies": ["direct"]}],
            "rules": ["MATCH,PROXY"],
        }
    )

    profile = await import_profile_from_yaml("My Profile", content)
    profiles = await list_profiles()

    assert profile["name"] == "My_Profile"
    assert profiles["current_profile"] == "My_Profile"
    assert profiles["profiles"][0]["source"] == "local"
    assert (get_paths().profiles / "My_Profile.yaml").is_file()


def test_mihomo_routes_are_registered():
    import web

    paths = web.app.openapi()["paths"]
    assert "/mihomo/status" in paths
    assert "/mihomo/config" in paths
    assert "/mihomo/proxies" in paths


@pytest.mark.asyncio
async def test_failed_start_restores_previous_enabled_state(monkeypatch):
    previous = MihomoSettings(enabled=False, profile="stable")
    saved = []

    async def fake_load_settings():
        return previous

    async def fake_save_settings(patch):
        saved.append(patch)
        return MihomoSettings.from_mapping({**previous.to_dict(), **patch})

    async def fake_start(_profile):
        raise FileNotFoundError("missing core")

    monkeypatch.setattr(panel_mihomo, "load_settings", fake_load_settings)
    monkeypatch.setattr(panel_mihomo, "save_settings", fake_save_settings)
    monkeypatch.setattr(panel_mihomo.mihomo_manager, "start", fake_start)

    with pytest.raises(HTTPException) as raised:
        await panel_mihomo.start_mihomo(
            panel_mihomo.MihomoStartRequest(profile="next"), _token="test"
        )

    assert raised.value.status_code == 400
    assert saved[-1] == {"enabled": False, "profile": "stable"}


@pytest.mark.asyncio
async def test_process_manager_rejects_multi_worker_ownership(tmp_path, monkeypatch):
    manager_module = importlib.import_module("src.mihomo.manager")

    async def fake_load_settings():
        return MihomoSettings(enabled=True, profile="default")

    monkeypatch.setattr(manager_module, "load_settings", fake_load_settings)
    monkeypatch.setenv("WORKERS", "2")
    monkeypatch.setenv("MIHOMO_DATA_DIR", str(tmp_path / "mihomo"))
    manager = manager_module.MihomoManager()

    with pytest.raises(ValueError, match="WORKERS=1"):
        await manager.start()
