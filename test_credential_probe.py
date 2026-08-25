from src.panel.creds import (
    ANTIGRAVITY_GEMINI_TEST_FALLBACK,
    credential_test_succeeded,
    select_antigravity_gemini_test_model,
)


def test_antigravity_probe_selects_lowest_text_gemini_quota():
    model, remaining = select_antigravity_gemini_test_model(
        {
            "models": {
                "gemini-3-pro-high": {"remaining": 0.42},
                "gemini-3-flash": {"remaining": 0},
                "gemini-3-pro-image": {"remaining": 0},
                "claude-sonnet-4-5": {"remaining": 1},
            }
        }
    )

    assert model == "gemini-3-flash"
    assert remaining == 0


def test_antigravity_probe_has_a_gemini_fallback_when_quota_is_unavailable():
    model, remaining = select_antigravity_gemini_test_model({"success": False, "models": {}})

    assert model == ANTIGRAVITY_GEMINI_TEST_FALLBACK
    assert remaining is None


def test_antigravity_429_is_not_reported_as_success():
    assert credential_test_succeeded("antigravity", 200) is True
    assert credential_test_succeeded("antigravity", 429) is False
    assert credential_test_succeeded("geminicli", 429) is True
