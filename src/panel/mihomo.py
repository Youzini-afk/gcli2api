"""Authenticated control-panel routes for Mihomo."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.mihomo.controller import MihomoControllerError
from src.mihomo.manager import mihomo_manager
from src.mihomo.profiles import (
    import_profile_from_url,
    import_profile_from_yaml,
    list_profiles,
    remove_profile,
)
from src.mihomo.settings import load_settings, save_settings
from src.mihomo.state import patch_state, read_state
from src.utils import verify_panel_token

router = APIRouter(prefix="/mihomo", tags=["mihomo"])


class MihomoConfigRequest(BaseModel):
    config: dict[str, Any]


class MihomoProfileUrlRequest(BaseModel):
    name: str = ""
    url: str


class MihomoProfileYamlRequest(BaseModel):
    name: str = ""
    content: str


class MihomoStartRequest(BaseModel):
    profile: str | None = None


class MihomoSelectRequest(BaseModel):
    name: str


def success(data: Any) -> JSONResponse:
    return JSONResponse(content={"success": True, "data": data})


def raise_controller_error(exc: MihomoControllerError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


def raise_lifecycle_error(exc: Exception) -> None:
    status_code = 400 if isinstance(exc, (FileNotFoundError, ValueError)) else 503
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get("/status")
async def get_status(_token: str = Depends(verify_panel_token)):
    return success(await mihomo_manager.status())


@router.put("/config")
async def update_config(request: MihomoConfigRequest, _token: str = Depends(verify_panel_token)):
    previous = await load_settings()
    current = await save_settings(request.config)
    status = await mihomo_manager.on_settings_updated(previous, current)
    return success({"config": current.to_dict(include_secret=False), "status": status})


@router.post("/start")
async def start_mihomo(request: MihomoStartRequest, _token: str = Depends(verify_panel_token)):
    previous = await load_settings()
    settings = await save_settings(
        {"enabled": True, **({"profile": request.profile} if request.profile else {})}
    )
    try:
        return success(await mihomo_manager.start(request.profile or settings.profile))
    except Exception as exc:
        await save_settings({"enabled": previous.enabled, "profile": previous.profile})
        raise_lifecycle_error(exc)


@router.post("/stop")
async def stop_mihomo(_token: str = Depends(verify_panel_token)):
    return success(await mihomo_manager.stop())


@router.post("/restart")
async def restart_mihomo(request: MihomoStartRequest, _token: str = Depends(verify_panel_token)):
    previous = await load_settings()
    was_running = mihomo_manager.is_running()
    await save_settings(
        {"enabled": True, **({"profile": request.profile} if request.profile else {})}
    )
    try:
        return success(await mihomo_manager.restart(request.profile))
    except Exception as exc:
        await save_settings(previous.to_dict())
        if was_running:
            try:
                await mihomo_manager.start(previous.profile)
            except Exception:
                pass
        raise_lifecycle_error(exc)


@router.get("/profiles")
async def get_profiles(_token: str = Depends(verify_panel_token)):
    return success(await list_profiles())


@router.post("/profiles/import-url")
async def import_url(request: MihomoProfileUrlRequest, _token: str = Depends(verify_panel_token)):
    try:
        previous = await load_settings()
        profile = await import_profile_from_url(request.name, request.url, previous)
        current = await save_settings({"profile": profile["name"]})
        try:
            status = await mihomo_manager.on_settings_updated(previous, current)
        except Exception as exc:
            await patch_state({"last_error": str(exc)})
            status = await mihomo_manager.status()
        return success({"profile": profile, "status": status})
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/profiles/import-yaml")
async def import_yaml(request: MihomoProfileYamlRequest, _token: str = Depends(verify_panel_token)):
    settings = await load_settings()
    if len(request.content.encode("utf-8")) > settings.max_subscription_size_bytes:
        raise HTTPException(status_code=413, detail="YAML 内容超过配置的订阅大小上限")
    try:
        previous = settings
        profile = await import_profile_from_yaml(request.name, request.content)
        current = await save_settings({"profile": profile["name"]})
        try:
            status = await mihomo_manager.on_settings_updated(previous, current)
        except Exception as exc:
            await patch_state({"last_error": str(exc)})
            status = await mihomo_manager.status()
        return success({"profile": profile, "status": status})
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/profiles/{name}")
async def delete_profile(name: str, _token: str = Depends(verify_panel_token)):
    settings = await load_settings()
    if settings.profile == name:
        raise HTTPException(status_code=400, detail="当前 Profile 不能删除，请先切换")
    await remove_profile(name)
    return success(await list_profiles())


@router.get("/proxies")
async def get_proxies(_token: str = Depends(verify_panel_token)):
    try:
        return success(await (await mihomo_manager.controller()).proxies())
    except MihomoControllerError as exc:
        raise_controller_error(exc)


@router.put("/proxies/{group}")
async def select_proxy(
    group: str,
    request: MihomoSelectRequest,
    _token: str = Depends(verify_panel_token),
):
    try:
        await (await mihomo_manager.controller()).select_proxy(group, request.name)
    except MihomoControllerError as exc:
        raise_controller_error(exc)
    state = await read_state()
    await patch_state({"selected": {**state["selected"], group: request.name}})
    return success({"group": group, "name": request.name})


@router.get("/proxies/{name}/delay")
async def test_delay(
    name: str,
    url: str | None = None,
    timeout: int = Query(default=5000, ge=1000, le=60000),
    _token: str = Depends(verify_panel_token),
):
    settings = await load_settings()
    try:
        result = await (await mihomo_manager.controller()).test_delay(
            name, url or settings.health_check_url, timeout
        )
        return success(result)
    except MihomoControllerError as exc:
        raise_controller_error(exc)
