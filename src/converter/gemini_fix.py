"""
Gemini Format Utilities - 统一的 Gemini 格式处理和转换工具
提供对 Gemini API 请求体和响应的标准化处理
────────────────────────────────────────────────────────────────
设计要点:
  * 每个模型在 MODEL_PROFILES 里有独立的一行配置 (思考类型 / 档位映射 / 是否禁预填充)
  * 模型名后缀 (-low / -high / -nothinking ...) 统一解析成 effort，再按模型各自的映射表落地
  * 客户端自带的 thinkingConfig 会被"翻译"成该模型能接受的形式 (budget <-> level 互转、越界钳制)
  * 不修改调用方传入的 request (generationConfig / thinkingConfig / tools / contents 均先复制)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from log import log
from src.utils import parse_model_feature

# ==================== Gemini API 配置 ====================

DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
]

MAX_OUTPUT_TOKENS = 64000
TOP_K = 64

# ==================== 每个模型的独立配置 ====================


@dataclass(frozen=True)
class ModelProfile:
    kind: str  # "budget" -> thinkingBudget, "level" -> thinkingLevel
    effort_map: Dict[str, Union[int, str]]  # effort -> 该模型实际使用的 budget / level
    budget_range: Tuple[int, int] = (0, 0)  # 仅 budget 模型: 客户端 budget 的钳制范围
    no_prefill: bool = False  # True: 请求不能以 model 消息结尾


# effort 取值: off / minimal / low / medium / high / max
# 无法真正关闭思考的模型, off 落到该模型的最低档
MODEL_PROFILES: Dict[str, ModelProfile] = {
    # ---------- Gemini 2.5: thinkingBudget ----------
    # Flash 可用 0 真正关闭
    "gemini-2.5-flash": ModelProfile(
        kind="budget",
        budget_range=(0, 24576),
        no_prefill=True,
        effort_map={
            "off": 0,
            "minimal": 0,
            "low": 1024,
            "medium": 8192,
            "high": 16000,
            "max": 24576,
        },
    ),
    # Pro 不能关闭, 最小 128
    "gemini-2.5-pro": ModelProfile(
        kind="budget",
        budget_range=(128, 32768),
        effort_map={
            "off": 128,
            "minimal": 128,
            "low": 1024,
            "medium": 8192,
            "high": 16000,
            "max": 32768,
        },
    ),
    # ---------- Gemini 3.x: thinkingLevel ----------
    # 3 Flash: 该渠道不支持 minimal (400), 仅 low / medium / high
    "gemini-3-flash-preview": ModelProfile(
        kind="level",
        no_prefill=True,
        effort_map={
            "off": "low",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
    # 3 Pro: 仅 low / high
    "gemini-3-pro-preview": ModelProfile(
        kind="level",
        effort_map={
            "off": "low",
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "max": "high",
        },
    ),
    # 3.1 Pro: low / medium / high
    "gemini-3.1-pro-preview": ModelProfile(
        kind="level",
        effort_map={
            "off": "low",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
    # 3.1 Flash-Lite: minimal / low / medium / high
    "gemini-3.1-flash-lite-preview": ModelProfile(
        kind="level",
        effort_map={
            "off": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
    # 3.1 Flash-Lite (gemini-3.1-flash-lite-preview 的别名, 配置完全相同)
    "gemini-3.1-flash-lite": ModelProfile(
        kind="level",
        effort_map={
            "off": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
    # 3.5 Flash: minimal / low / medium / high, 默认 medium, 不支持预填充
    "gemini-3.5-flash": ModelProfile(
        kind="level",
        no_prefill=True,
        effort_map={
            "off": "minimal",
            "minimal": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
    # 3.8 Flash: 仅 low / medium / high (minimal 会 400), 默认 medium, 不支持预填充
    "gemini-3.8-flash": ModelProfile(
        kind="level",
        no_prefill=True,
        effort_map={
            "off": "low",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "max": "high",
        },
    ),
}

# ==================== 模型名解析 ====================


def parse_model_name(model_name: str) -> Tuple[str, Optional[str], bool]:
    """
    解析模型名 -> (基础模型名, effort, 是否搜索)
    例: gemini-2.5-pro-high-search -> ("gemini-2.5-pro", "high", True)
    """
    feature = parse_model_feature(model_name, backend="geminicli")
    effort = {"nothinking": "off", "maxthinking": "max", "think": None}.get(
        feature.legacy_thinking, feature.legacy_thinking
    )
    return feature.base_model, effort, feature.search_enabled


# ==================== 思考配置解析 ====================

_LEVEL_TO_EFFORT = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _budget_to_effort(budget: int) -> Optional[str]:
    """客户端 budget -> effort (用于 level 模型); 负数(动态)返回 None = 用模型默认"""
    if budget < 0:
        return None
    if budget == 0:
        return "off"
    if budget <= 1024:
        return "low"
    if budget <= 8192:
        return "medium"
    if budget <= 16000:
        return "high"
    return "max"


def _apply_effort(profile: ModelProfile, effort: str) -> Tuple[str, Union[int, str]]:
    key = "thinkingBudget" if profile.kind == "budget" else "thinkingLevel"
    return key, profile.effort_map[effort]


def resolve_thinking(
    profile: ModelProfile,
    suffix_effort: Optional[str],
    client_cfg: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, Union[int, str]]]:
    """
    决定最终的思考参数, 优先级: 客户端 thinkingConfig > 模型名后缀 > 模型默认
    返回 (参数名, 值); None 表示不设置, 由模型使用默认思考
    """
    if client_cfg is None:
        return _apply_effort(profile, suffix_effort) if suffix_effort else None

    budget = client_cfg.get("thinkingBudget")
    level = client_cfg.get("thinkingLevel")
    level = level.lower() if isinstance(level, str) else None

    if profile.kind == "budget":
        if _is_int(budget):
            if budget < 0:
                return "thinkingBudget", -1  # 动态思考
            lo, hi = profile.budget_range
            return "thinkingBudget", min(max(budget, lo), hi)
        if level in _LEVEL_TO_EFFORT:
            return _apply_effort(profile, level)
    else:
        if level in _LEVEL_TO_EFFORT:
            return _apply_effort(profile, _LEVEL_TO_EFFORT[level])
        if _is_int(budget):
            effort = _budget_to_effort(budget)
            if effort:
                return _apply_effort(profile, effort)
    return None


# ==================== 兼容旧接口 ====================


def get_base_model_name(model_name: str) -> str:
    """移除模型名称中的后缀,返回基础模型名"""
    return parse_model_name(model_name)[0]


def get_thinking_settings(model_name: str) -> tuple[Optional[int], Optional[str]]:
    """仅按模型名后缀返回 (thinking_budget, thinking_level), 无后缀或未知模型返回 (None, None)"""
    base, effort, _ = parse_model_name(model_name)
    profile = MODEL_PROFILES.get(base)
    if not profile or not effort:
        return None, None
    value = profile.effort_map[effort]
    return (value, None) if profile.kind == "budget" else (None, value)


def is_search_model(model_name: str) -> bool:
    """检查是否为搜索模型"""
    return parse_model_feature(model_name, backend="geminicli").search_enabled


def is_thinking_model(model_name: str) -> bool:
    """检查是否为思考模型 (旧逻辑, 仅为兼容保留)"""
    feature = parse_model_feature(model_name, backend="geminicli")
    return bool(
        feature.legacy_thinking or "think" in feature.base_model or "pro" in feature.base_model
    )


# ==================== 请求体清理 ====================


def _clean_contents(contents: List[Any]) -> List[Any]:
    """过滤空 part、修正 text 字段类型、丢弃没有有效 part 的 content"""
    cleaned_contents = []
    for content in contents:
        if not (isinstance(content, dict) and "parts" in content):
            cleaned_contents.append(content)
            continue

        valid_parts = []
        for part in content["parts"]:
            if not isinstance(part, dict):
                continue

            part = part.copy()
            if "text" in part:
                text_value = part["text"]
                if isinstance(text_value, list):
                    log.warning(f"[GEMINI_FIX] text 字段是列表，自动合并: {text_value}")
                    part["text"] = " ".join(
                        str(t.get("text", "")) if isinstance(t, dict) else str(t)
                        for t in text_value
                        if t is not None
                    ).rstrip()
                elif isinstance(text_value, str):
                    part["text"] = text_value.rstrip()
                else:
                    log.warning(
                        f"[GEMINI_FIX] text 字段类型异常 ({type(text_value)}), 转为字符串: {text_value}"
                    )
                    part["text"] = str(text_value)
                if not part["text"]:
                    part.pop("text", None)
            # Normalize first: whitespace-only text must not become an empty
            # upstream text block after it has already passed validation.
            if not any(
                value not in (None, "", {}, []) for key, value in part.items() if key != "thought"
            ):
                log.warning(f"[GEMINI_FIX] 移除空的或无效的 part: {part}")
                continue
            valid_parts.append(part)

        if valid_parts:
            cleaned_content = content.copy()
            cleaned_content["parts"] = valid_parts
            cleaned_contents.append(cleaned_content)
        else:
            log.warning(f"[GEMINI_FIX] 跳过没有有效 parts 的 content: {content.get('role')}")
    return cleaned_contents


def _strip_trailing_model_turns(contents: List[Any]) -> Tuple[List[Any], int]:
    """循环移除末尾的 model 消息，保证以用户消息结尾"""
    contents = list(contents)
    removed = 0
    while contents and isinstance(contents[-1], dict) and contents[-1].get("role") == "model":
        contents.pop()
        removed += 1
    return contents, removed


# ==================== 统一的 Gemini 请求后处理 ====================


async def normalize_gemini_request(
    request: Dict[str, Any],
    mode: str = "geminicli",
    preserve_trailing_model: bool = False,
) -> Dict[str, Any]:
    """
    规范化 Gemini 请求

    处理逻辑:
    1. 解析模型名后缀 -> 基础模型 / effort / 搜索
    2. 按模型 profile 生成 thinkingConfig (budget 与 level 互斥, 自动翻译/钳制)
    3. 搜索模型添加 googleSearch 工具
    4. 按模型特性处理预填充
    5. 公共参数与 contents 清理

    Args:
        request: 原始请求字典 (不会被修改)
        mode: 模式 ("geminicli")

    Returns:
        规范化后的请求
    """
    from config import get_return_thoughts_to_frontend

    result = request.copy()
    model = result.get("model", "")
    generation_config = (result.get("generationConfig") or {}).copy()

    log.debug(
        f"[GEMINI_FIX] 原始请求 - 模型: {model}, mode: {mode}, generationConfig: {generation_config}"
    )

    base_model, suffix_effort, search = parse_model_name(model)
    profile = MODEL_PROFILES.get(base_model)

    # ========== 1. 思考设置 (按模型独立处理) ==========
    if profile:
        return_thoughts = await get_return_thoughts_to_frontend()

        explicit = isinstance(generation_config.get("thinkingConfig"), dict)
        thinking_config = dict(generation_config.get("thinkingConfig") or {})
        client_cfg = dict(thinking_config) if explicit else None
        # 先清掉两个互斥字段, 再写入该模型能接受的那一个
        thinking_config.pop("thinkingBudget", None)
        thinking_config.pop("thinkingLevel", None)

        resolved = resolve_thinking(profile, suffix_effort, client_cfg)
        thinking_off = False
        if resolved:
            key, value = resolved
            thinking_config[key] = value
            thinking_off = key == "thinkingBudget" and value == 0

        # 只有 budget=0 (真正关闭) 时不返回思考
        if thinking_off:
            thinking_config["includeThoughts"] = False
        else:
            thinking_config.setdefault("includeThoughts", return_thoughts)
        generation_config["thinkingConfig"] = thinking_config
    elif suffix_effort:
        log.warning(f"[GEMINI_FIX] 未知模型 {base_model}，忽略思考后缀 (effort={suffix_effort})")

    # ========== 2. 搜索模型添加 Google Search ==========
    if search or is_search_model(model):
        result_tools = list(result.get("tools") or [])
        if not any("googleSearch" in tool for tool in result_tools if isinstance(tool, dict)):
            result_tools.append({"googleSearch": {}})
        result["tools"] = result_tools

    # ========== 3. 模型名称处理 ==========
    result["model"] = base_model

    # ========== 公共处理 ==========
    generation_config.pop("presencePenalty", None)
    generation_config.pop("frequencyPenalty", None)
    generation_config.pop("stopSequences", None)

    result["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    if generation_config:
        generation_config["maxOutputTokens"] = MAX_OUTPUT_TOKENS
        generation_config["topK"] = TOP_K
        result["generationConfig"] = generation_config

    # contents: 先清理, 再处理预填充 (清理后可能产生新的末尾 model 消息)
    if "contents" in result:
        contents = _clean_contents(result["contents"])
        if profile and profile.no_prefill and not preserve_trailing_model:
            contents, removed = _strip_trailing_model_turns(contents)
            if removed:
                log.warning(
                    f"[GEMINI_FIX] {base_model} 不支持预填充，移除了 {removed} 条末尾 model 消息"
                )
        result["contents"] = contents

    return result
