import re
from typing import Any, Dict, List, Optional

from config import get_api_password, get_panel_password
from fastapi import Depends, HTTPException, Header, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from log import log
from src.models import ModelFeature

# HTTP Bearer security scheme. Panel auth handles missing credentials itself so
# its failures can carry an explicit marker for the browser.
security = HTTPBearer(auto_error=False)
PANEL_AUTH_FAILURE_HEADER = "X-Panel-Auth-Failed"

# ====================== OAuth Configuration ======================

def get_geminicli_user_agent(model: str = "") -> str:
    """生成上游 Gemini CLI 的浏览器兼容 User-Agent。"""
    if model:
        return f"Mozilla/5.0 (compatible; Google-Gemini-CLI/1.0; +https://github.com/google-gemini/gemini-cli) {model}"
    return "Mozilla/5.0 (compatible; Google-Gemini-CLI/1.0; +https://github.com/google-gemini/gemini-cli)"


# 静态常量
GEMINICLI_USER_AGENT = get_geminicli_user_agent()

# Antigravity CLI 客户端仿真常量
ANTIGRAVITY_CLI_VERSION = "2.22.0"
ANTIGRAVITY_CLI_PLATFORM = "darwin/arm64"
ANTIGRAVITY_USER_AGENT = f"antigravity/hub/{ANTIGRAVITY_CLI_VERSION} {ANTIGRAVITY_CLI_PLATFORM}"

# OAuth Configuration - 标准模式
CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
CLIENT_SECRET = "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"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# Antigravity OAuth Configuration
ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# 统一的 Token URL（两种模式相同）
TOKEN_URL = "https://oauth2.googleapis.com/token"

# 回调服务器配置
CALLBACK_HOST = "localhost"

# Model name lists for different features
BASE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]


# ====================== Model Helper Functions ======================

_MODEL_TRANSPORT_PREFIXES = (
    ("假流式/", "fake_stream"),
    ("流式抗截断/", "stream_anti_truncation"),
    ("工具抗截断/", "tool_anti_truncation"),
    ("抗截断/", "tool_anti_truncation"),
)

# These are client-facing compatibility suffixes.  They are intentionally
# parsed as names (rather than matching arbitrary text in a model ID).
_LEGACY_THINKING_SUFFIXES = (
    "maxthinking",
    "nothinking",
    "minimal",
    "medium",
    "high",
    "low",
    "max",
    "think",
)

_EXPERIMENTAL_MARKER_RE = re.compile(r"\s*\[(混合实验性|实验性)\]\s*", re.UNICODE)


def _strip_model_markers(model_name: str) -> tuple[str, Optional[str]]:
    """Remove experimental labels while retaining the model ID itself."""
    marker_names = _EXPERIMENTAL_MARKER_RE.findall(model_name)
    if not marker_names:
        return model_name.strip(), None

    # A label is metadata and may occur before or after a transport prefix.
    # Removing all occurrences also accepts both ``[实验性]model`` and
    # ``model [实验性]`` forms (with or without a separating space).
    result = _EXPERIMENTAL_MARKER_RE.sub("", model_name).strip()
    # Be liberal about the common ``[实验性]/model`` spelling.
    result = result.lstrip("/").strip()
    marker = "混合实验性" if "混合实验性" in marker_names else "实验性"
    return result, marker


def parse_model_feature(model_name: str, *, backend: str | None = None) -> ModelFeature:
    """Parse transport, experiment and legacy model-name features once.

    ``backend="antigravity"`` is deliberately conservative: ``-high`` and
    ``-low`` (also ``-medium`` for Gemini 3) can be real upstream model IDs,
    so those suffixes are not removed there. Other compatibility suffixes remain
    available for old callers.  GeminiCLI treats all known thinking suffixes
    as compatibility metadata.
    """
    original = "" if model_name is None else str(model_name)
    marked_name, marker = _strip_model_markers(original)

    transport = "normal"
    remainder = marked_name
    for prefix, transport_name in _MODEL_TRANSPORT_PREFIXES:
        if remainder.startswith(prefix):
            transport = transport_name
            remainder = remainder[len(prefix) :].lstrip()
            break

    request_model = remainder.strip()
    if marker == "混合实验性":
        transport = "image_current_turn"
    elif marker == "实验性":
        transport = "image_all"

    # Search can be combined with thinking (e.g. ``-high-search``), so peel it
    # first.  Matching at the end prevents model IDs containing the word
    # "search" from being altered.
    search_enabled = False
    remainder = remainder.strip()
    while remainder.lower().endswith("-search"):
        search_enabled = True
        remainder = remainder[: -len("-search")].rstrip()

    legacy_thinking: Optional[str] = None
    lowered = remainder.lower()
    for suffix in _LEGACY_THINKING_SUFFIXES:
        token = f"-{suffix}"
        if not lowered.endswith(token):
            continue
        # Antigravity's high/low IDs are ambiguous.  Only the unambiguous
        # compatibility suffixes are interpreted on that backend.
        if backend == "antigravity" and (
            suffix in {"high", "low"}
            or (suffix == "medium" and lowered.startswith("gemini-3"))
        ):
            break
        remainder = remainder[: -len(token)].rstrip()
        legacy_thinking = suffix
        break

    return ModelFeature(
        original_model=original,
        base_model=remainder,
        request_model=request_model,
        transport=transport,
        legacy_thinking=legacy_thinking,
        search_enabled=search_enabled,
        experimental=marker is not None,
        mixed_experimental=marker == "混合实验性",
        experimental_marker=marker,
    )


def is_fake_streaming_model(model_name: str) -> bool:
    """Check if model name indicates fake streaming should be used."""
    return parse_model_feature(model_name).transport == "fake_stream"


def is_anti_truncation_model(model_name: str) -> bool:
    """Check if model name indicates anti-truncation should be used."""
    return parse_model_feature(model_name).transport == "stream_anti_truncation"


def is_tool_anti_truncation_model(model_name: str) -> bool:
    """Check if synthetic-tool anti-truncation should be used."""
    return parse_model_feature(model_name).transport == "tool_anti_truncation"


def get_base_model_from_feature_model(model_name: str) -> str:
    """Remove only transport/experiment markers (historical route behavior).

    Legacy thinking/search suffixes are intentionally retained.  The request
    converters need to see them in order to provide fallback behavior, and
    Antigravity must not accidentally lose a real ``-high``/``-low`` ID.
    """
    return parse_model_feature(model_name).request_model


def apply_model_feature_flags(api_request: Dict[str, Any], feature: ModelFeature) -> Dict[str, Any]:
    """Attach private transport controls to one normalized API request."""
    if feature.transport == "tool_anti_truncation":
        from src.converter.tool_anti_truncation import TOOL_ANTI_TRUNCATION_FLAG

        api_request[TOOL_ANTI_TRUNCATION_FLAG] = True
    elif feature.transport in {"image_all", "image_current_turn"}:
        from src.converter.image_context import IMAGE_CONTEXT_MODE_FLAG, ImageContextMode

        api_request[IMAGE_CONTEXT_MODE_FLAG] = (
            ImageContextMode.ALL_TEXT.value
            if feature.transport == "image_all"
            else ImageContextMode.CURRENT_TURN.value
        )
    return api_request


def thinking_config_for_effort(model_name: str, effort: object) -> dict:
    """Translate an explicit OpenAI reasoning effort to Gemini config.

    The mapping mirrors the existing GeminiCLI compatibility suffix mapping.
    For an unknown model family, a recognized level is still expressed via
    ``thinkingLevel``; no model ID is invented or rewritten here.
    """
    if effort is None:
        return {}
    value = str(effort).strip().lower()
    if not value:
        return {}

    if value in {"none", "off", "disabled", "disable"}:
        return {"includeThoughts": False}

    feature = parse_model_feature(model_name, backend="geminicli")
    base = feature.base_model.lower()
    if value == "max":
        if "gemini-2.5" in base:
            return {"thinkingBudget": 24576 if "flash" in base else 32768}
        return {"thinkingLevel": "high"}

    if value not in {"minimal", "low", "medium", "high"}:
        return {}

    if "gemini-2.5" in base:
        budgets = {
            "minimal": 0 if "flash" in base else 128,
            "low": 1024,
            "medium": 8192,
            "high": 16000,
        }
        return {"thinkingBudget": budgets[value]}

    # Gemini 3 and newer routes use thinkingLevel.  Keeping the value as a
    # level also gives unknown Antigravity model IDs the best explicit effort
    # behavior without fabricating a route name.
    return {"thinkingLevel": value}


# Descriptive alias for callers that prefer the ``get_*`` naming convention.
get_thinking_config_for_effort = thinking_config_for_effort


def get_available_models(router_type: str = "openai") -> List[str]:
    """
    Get available models with feature prefixes.

    Args:
        router_type: "openai" or "gemini"

    Returns:
        List of model names with feature prefixes
    """
    models = []

    for base_model in BASE_MODELS:
        # 基础模型
        models.append(base_model)

        # 假流式模型 (前缀格式)
        models.append(f"假流式/{base_model}")

        # 流式续写抗截断（保留独立前缀）
        models.append(f"流式抗截断/{base_model}")

        # 合成工具抗截断模型（支持流式与非流式）
        models.append(f"工具抗截断/{base_model}")

        # 图片上下文模型（下行自动复用工具抗截断）
        models.append(f"[实验性] {base_model}")
        models.append(f"[混合实验性] {base_model}")

    return models


# ====================== Authentication Functions ======================


async def authenticate_flexible(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    access_token: Optional[str] = Header(None, alias="access_token"),
    x_goog_api_key: Optional[str] = Header(None, alias="x-goog-api-key"),
    x_anthropic_auth_token: Optional[str] = Header(None, alias="x-anthropic-auth-token"),
    anthropic_auth_token: Optional[str] = Header(None, alias="anthropic-auth-token"),
    key: Optional[str] = Query(None),
) -> str:
    """
    统一的灵活认证函数，支持多种认证方式

    此函数可以直接用作 FastAPI 的 Depends 依赖

    支持的认证方式:
        - URL 参数: key
        - HTTP 头部: Authorization (Bearer token)
        - HTTP 头部: x-api-key
        - HTTP 头部: access_token
        - HTTP 头部: x-goog-api-key
        - HTTP 头部: x-anthropic-auth-token
        - HTTP 头部: anthropic-auth-token

    Args:
        request: FastAPI Request 对象
        authorization: Authorization 头部值（自动注入）
        x_api_key: x-api-key 头部值（自动注入）
        access_token: access_token 头部值（自动注入）
        x_goog_api_key: x-goog-api-key 头部值（自动注入）
        x_anthropic_auth_token: x-anthropic-auth-token 头部值（自动注入）
        anthropic_auth_token: anthropic-auth-token 头部值（自动注入）
        key: URL 参数 key（自动注入）

    Returns:
        验证通过的token

    Raises:
        HTTPException: 认证失败时抛出异常

    使用示例:
        @router.post("/endpoint")
        async def endpoint(token: str = Depends(authenticate_flexible)):
            # token 已验证通过
            pass
    """
    password = await get_api_password()
    token = None
    auth_method = None

    # 1. 尝试从 URL 参数 key 获取（Google 官方标准方式）
    if key:
        token = key
        auth_method = "URL parameter 'key'"

    # 2. 尝试从 x-goog-api-key 头部获取（Google API 标准方式）
    elif x_goog_api_key:
        token = x_goog_api_key
        auth_method = "x-goog-api-key header"

    # 3. 尝试从 x-anthropic-auth-token 头部获取（Anthropic 标准方式）
    elif x_anthropic_auth_token:
        token = x_anthropic_auth_token
        auth_method = "x-anthropic-auth-token header"

    # 4. 尝试从 anthropic-auth-token 头部获取（Anthropic 替代方式）
    elif anthropic_auth_token:
        token = anthropic_auth_token
        auth_method = "anthropic-auth-token header"

    # 5. 尝试从 x-api-key 头部获取
    elif x_api_key:
        token = x_api_key
        auth_method = "x-api-key header"

    # 6. 尝试从 access_token 头部获取
    elif access_token:
        token = access_token
        auth_method = "access_token header"

    # 7. 尝试从 Authorization 头部获取
    elif authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme. Use 'Bearer <token>'",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization[7:]  # 移除 "Bearer " 前缀
        auth_method = "Authorization Bearer header"

    # 检查是否提供了任何认证凭据
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials. Use 'key' URL parameter, 'x-goog-api-key', 'x-anthropic-auth-token', 'anthropic-auth-token', 'x-api-key', 'access_token' header, or 'Authorization: Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证 token
    if token != password:
        log.debug(f"Authentication failed using {auth_method}")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="密码错误")

    log.debug(f"Authentication successful using {auth_method}")
    return token


# 为了保持向后兼容，保留旧函数名作为别名
authenticate_bearer = authenticate_flexible
authenticate_gemini_flexible = authenticate_flexible


# ====================== Panel Authentication Functions ======================


async def verify_panel_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """
    简化的控制面板密码验证函数

    直接验证Bearer token是否等于控制面板密码

    Args:
        credentials: HTTPAuthorizationCredentials 自动注入

    Returns:
        验证通过的token

    Raises:
        HTTPException: 密码错误时抛出401异常
    """

    password = await get_panel_password()
    if credentials is None or credentials.credentials != password:
        raise HTTPException(
            status_code=401,
            detail="密码错误",
            headers={PANEL_AUTH_FAILURE_HEADER: "1"},
        )
    return credentials.credentials
