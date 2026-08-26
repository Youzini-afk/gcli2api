"""
根路由模块 - 处理控制面板主页
"""

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from log import log

# 创建路由器
router = APIRouter(tags=["root"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONT_DIR = PROJECT_ROOT / "front"


def get_control_panel_asset_version() -> str:
    """Hash UI assets so every deployment gets cache-safe URLs."""
    digest = hashlib.sha256()
    for filename in ("control-panel.css", "control-panel.js"):
        digest.update((FRONT_DIR / filename).read_bytes())
    return digest.hexdigest()[:12]


@router.get("/", response_class=HTMLResponse)
async def serve_control_panel():
    """提供统一控制面板"""
    try:
        with (FRONT_DIR / "console.html").open("r", encoding="utf-8") as f:
            html_content = f.read()
        html_content = html_content.replace("__ASSET_VERSION__", get_control_panel_asset_version())
        return HTMLResponse(
            content=html_content,
            headers={"Cache-Control": "no-store"},
        )

    except Exception as e:
        log.error(f"加载控制面板页面失败: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")
