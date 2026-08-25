"""
根路由模块 - 处理控制面板主页
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from log import log


# 创建路由器
router = APIRouter(tags=["root"])


@router.get("/", response_class=HTMLResponse)
async def serve_control_panel():
    """提供统一控制面板"""
    try:
        html_file_path = "front/console.html"

        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    except Exception as e:
        log.error(f"加载控制面板页面失败: {e}")
        raise HTTPException(status_code=500, detail="服务器内部错误")
