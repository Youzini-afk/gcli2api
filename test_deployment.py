from pathlib import Path

import pytest
import yaml

from src.panel.root import serve_control_panel

PROJECT_ROOT = Path(__file__).resolve().parent


def test_zeabur_template_builds_from_this_repository():
    template = yaml.safe_load((PROJECT_ROOT / "zeabur.yaml").read_text(encoding="utf-8"))
    service = template["spec"]["services"][0]

    assert service["template"] == "GIT"
    assert service["spec"]["source"] == {
        "source": "GITHUB",
        "repo": 1346064576,
        "branch": "master",
    }


@pytest.mark.asyncio
async def test_control_panel_assets_are_versioned_and_html_is_not_cached():
    response = await serve_control_panel()
    html = response.body.decode("utf-8")

    assert "__ASSET_VERSION__" not in html
    assert "/front/control-panel.js?v=" in html
    assert "/front/control-panel.css?v=" in html
    assert response.headers["cache-control"] == "no-store"
