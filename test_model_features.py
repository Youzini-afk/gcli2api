import pytest

from src.converter.antigravity_fix import normalize_antigravity_request
from src.converter.gemini_fix import normalize_gemini_request
from src.converter.openai2gemini import convert_openai_to_gemini_request
from src.converter.anthropic2gemini import anthropic_to_gemini_request
from src.models import ModelFeature
from src.utils import (
    get_available_models,
    get_base_model_from_feature_model,
    parse_model_feature,
)


def test_model_feature_parses_transport_markers_and_legacy_suffixes():
    feature = parse_model_feature("假流式/[实验性] gemini-2.5-flash-high-search")

    assert isinstance(feature, ModelFeature)
    # Image-context markers are a complete transport mode and take precedence
    # over another legacy transport prefix if clients accidentally combine them.
    assert feature.transport == "image_all"
    assert feature.experimental is True
    assert feature.mixed_experimental is False
    assert feature.legacy_thinking == "high"
    assert feature.search_enabled is True
    assert feature.base_model == "gemini-2.5-flash"

    mixed = parse_model_feature("[混合实验性]工具抗截断/gemini-3-flash-preview-medium")
    assert mixed.transport == "image_current_turn"
    assert mixed.experimental_marker == "混合实验性"
    assert mixed.mixed_experimental is True
    assert mixed.base_model == "gemini-3-flash-preview"


def test_compatibility_wrapper_only_removes_transport_marker():
    assert (
        get_base_model_from_feature_model("流式抗截断/gemini-3-pro-high-search")
        == "gemini-3-pro-high-search"
    )


@pytest.mark.asyncio
async def test_geminicli_explicit_thinking_config_wins_over_suffix():
    result = await normalize_gemini_request(
        {
            "model": "gemini-2.5-flash-high",
            "generationConfig": {"thinkingConfig": {"thinkingBudget": 777}},
        }
    )

    assert result["model"] == "gemini-2.5-flash"
    assert result["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 777


@pytest.mark.asyncio
async def test_openai_reasoning_effort_is_explicit_thinking_config():
    result = await convert_openai_to_gemini_request(
        {
            "model": "gemini-2.5-flash-high",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning": {"effort": "low"},
        }
    )

    assert result["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}

    disabled = await convert_openai_to_gemini_request(
        {
            "model": "gemini-2.5-flash-high",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "none",
        }
    )
    assert disabled["generationConfig"]["thinkingConfig"] == {"includeThoughts": False}

    future_value = await convert_openai_to_gemini_request(
        {
            "model": "gemini-2.5-flash-high",
            "messages": [{"role": "user", "content": "hello"}],
            "reasoning_effort": "future-level",
        }
    )
    normalized = await normalize_gemini_request({"model": "gemini-2.5-flash-high", **future_value})
    assert "thinkingBudget" not in normalized["generationConfig"]["thinkingConfig"]


@pytest.mark.asyncio
async def test_anthropic_thinking_is_explicit_even_when_disabled():
    result = await anthropic_to_gemini_request(
        {
            "model": "gemini-2.5-flash-high",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "disabled"},
        }
    )

    assert result["generationConfig"]["thinkingConfig"] == {"includeThoughts": False}

    adaptive = await anthropic_to_gemini_request(
        {
            "model": "gemini-2.5-flash-medium",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
        }
    )
    normalized = await normalize_gemini_request({"model": "gemini-2.5-flash-medium", **adaptive})
    assert "thinkingBudget" not in normalized["generationConfig"]["thinkingConfig"]


@pytest.mark.asyncio
async def test_antigravity_preserves_real_high_low_ids_and_explicit_effort():
    real_id = await normalize_antigravity_request(
        {"model": "gemini-3-pro-high", "generationConfig": {}}
    )
    assert real_id["model"] == "gemini-3-pro-high"

    explicit = await normalize_antigravity_request(
        {
            "model": "gemini-3-pro-preview",
            "generationConfig": {"thinkingConfig": {"thinkingLevel": "low"}},
        }
    )
    assert explicit["model"] == "gemini-3-pro-preview"
    assert explicit["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"


@pytest.mark.asyncio
async def test_antigravity_known_pro_high_alias_keeps_reasoning_output_setting():
    result = await normalize_antigravity_request(
        {"model": "gemini-3.1-pro-high", "generationConfig": {}}
    )

    assert result["model"] == "gemini-pro-agent"
    assert "includeThoughts" in result["generationConfig"]["thinkingConfig"]

    with_search = await normalize_antigravity_request(
        {"model": "gemini-3.1-pro-high-search", "generationConfig": {}}
    )
    assert with_search["model"] == "gemini-pro-agent"
    assert {"googleSearch": {}} in with_search["tools"]


def test_geminicli_model_list_does_not_expand_thinking_or_search():
    models = get_available_models("gemini")

    assert len(models) == 36
    assert all("-search" not in model for model in models)
    assert all(
        not model.endswith(("-high", "-medium", "-low", "-minimal", "-max")) for model in models
    )


@pytest.mark.asyncio
async def test_antigravity_model_list_adds_only_transport_variants(monkeypatch):
    pytest.importorskip("PIL")
    from src.router.antigravity.model_list import get_antigravity_models_with_features

    async def fake_fetch_available_models():
        return [
            {"id": "gemini-3-pro-high"},
            {"id": "gemini-3-pro-low"},
            {"id": "gemini-3.1-flash-image"},
        ]

    import src.router.antigravity.model_list as model_list

    monkeypatch.setattr(model_list, "fetch_available_models", fake_fetch_available_models)
    models = await get_antigravity_models_with_features()

    assert models == [
        "gemini-3-pro-high",
        "假流式/gemini-3-pro-high",
        "流式抗截断/gemini-3-pro-high",
        "工具抗截断/gemini-3-pro-high",
        "[实验性] gemini-3-pro-high",
        "[混合实验性] gemini-3-pro-high",
        "gemini-3-pro-low",
        "假流式/gemini-3-pro-low",
        "流式抗截断/gemini-3-pro-low",
        "工具抗截断/gemini-3-pro-low",
        "[实验性] gemini-3-pro-low",
        "[混合实验性] gemini-3-pro-low",
        "gemini-3.1-flash-image",
        "假流式/gemini-3.1-flash-image",
        "流式抗截断/gemini-3.1-flash-image",
        "工具抗截断/gemini-3.1-flash-image",
    ]
