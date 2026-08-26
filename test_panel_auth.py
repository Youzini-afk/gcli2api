import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import src.utils as utils


@pytest.fixture(autouse=True)
def panel_password(monkeypatch):
    async def fake_panel_password():
        return "panel-secret"

    monkeypatch.setattr(utils, "get_panel_password", fake_panel_password)


@pytest.mark.asyncio
async def test_valid_panel_token_is_accepted():
    credentials = HTTPAuthorizationCredentials(
        scheme="Bearer",
        credentials="panel-secret",
    )

    assert await utils.verify_panel_token(credentials) == "panel-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credentials",
    [
        None,
        HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-secret"),
    ],
)
async def test_panel_auth_failure_has_an_explicit_browser_marker(credentials):
    with pytest.raises(HTTPException) as raised:
        await utils.verify_panel_token(credentials)

    assert raised.value.status_code == 401
    assert raised.value.headers == {utils.PANEL_AUTH_FAILURE_HEADER: "1"}
