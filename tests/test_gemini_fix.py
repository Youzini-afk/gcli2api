from copy import deepcopy

import pytest

from src.converter.antigravity_fix import (
    _ensure_empty_tool_schema_for_claude,
    normalize_antigravity_request,
)
from src.converter.gemini_fix import normalize_gemini_request


def test_antigravity_claude_tools_keep_schema_in_parameters():
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parametersJsonSchema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                }
            ]
        }
    ]

    result = _ensure_empty_tool_schema_for_claude(tools, "claude-opus-4-6-thinking", "antigravity")
    declaration = result[0]["functionDeclarations"][0]

    assert declaration["parameters"]["type"] == "object"
    assert "parametersJsonSchema" not in declaration


@pytest.mark.asyncio
async def test_claude_normalization_keeps_all_tools_explicit_thinking_and_call_ids():
    request = {
        "model": "claude-opus-4-6-thinking",
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 4096, "includeThoughts": False}},
        "tools": [
            {
                "functionDeclarations": [
                    {"name": name, "parameters": {"type": "object", "properties": {}}}
                    for name in ("read", "write")
                ]
            },
            {"googleSearch": {}},
        ],
        "contents": [
            {"role": "user", "parts": [{"text": "read a file"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"id": "original", "name": "read", "args": {}}}],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "read", "response": {"result": "ok"}}}],
            },
        ],
    }
    original = deepcopy(request)
    result = await normalize_antigravity_request(request)

    assert [tool["name"] for tool in result["tools"][0]["functionDeclarations"]] == [
        "read",
        "write",
    ]
    assert result["tools"][1] == {"googleSearch": {}}
    assert (
        result["generationConfig"]["thinkingConfig"]
        == original["generationConfig"]["thinkingConfig"]
    )
    assert result["contents"][-1]["parts"][0]["functionResponse"]["id"] == "original"
    assert request == original


@pytest.mark.asyncio
@pytest.mark.parametrize("normalize", [normalize_gemini_request, normalize_antigravity_request])
async def test_normalization_cleans_blank_text_before_validating_parts(normalize):
    request = {
        "model": "gemini-3.8-flash",
        "contents": [
            {"role": "user", "parts": [{"text": [{"type": "text", "text": "hello"}]}]},
            {"role": "model", "parts": [{"text": " \n"}]},
        ],
    }
    original = deepcopy(request)
    result = await normalize(request)
    assert result["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert request == original
