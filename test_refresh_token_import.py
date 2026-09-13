import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import UploadFile

import src.panel.creds as creds


def _response_body(response):
    return json.loads(response.body)


class _FakeStorage:
    def __init__(self):
        self.stored = []
        self.states = []

    async def store_credential(self, filename, data, mode="geminicli"):
        self.stored.append((filename, data, mode))
        return True

    async def update_credential_state(self, filename, state, mode="geminicli"):
        self.states.append((filename, state, mode))
        return True

    class _Backend:
        async def clear_all_model_cooldowns(self, filename, mode="geminicli"):
            return True

    _backend = _Backend()


def _upload(filename, payload):
    return UploadFile(
        filename=filename,
        file=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


@pytest.mark.asyncio
async def test_upload_recognizes_nested_refresh_token_array_and_deduplicates(monkeypatch):
    imported = []

    async def fake_import(refresh_token, email=None, project_id=None, mode="geminicli"):
        imported.append((refresh_token, email, project_id, mode))
        return {"filename": f"rt_{refresh_token}.json", "status": "success", "message": "导入成功"}

    monkeypatch.setattr(creds, "import_refresh_token_item", fake_import)
    payload = [
        {"email": "one@example.com", "refresh_token": "rt-one"},
        {"nested": {"refreshToken": "rt-two", "email": "two@example.com"}},
        {"email": "duplicate@example.com", "refresh_token": "rt-one"},
    ]

    response = await creds.upload_credentials_common([_upload("export.json", payload)])
    body = _response_body(response)

    assert body["imported_count"] == 2
    assert body["total_count"] == 2
    assert [item[0] for item in imported] == ["rt-one", "rt-two"]


@pytest.mark.asyncio
async def test_direct_refresh_token_import_uses_mode_specific_configuration(monkeypatch):
    class FakeCredentials:
        def __init__(self, access_token, refresh_token, client_id, client_secret):
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.client_id = client_id
            self.client_secret = client_secret
            self.expires_at = None

        async def refresh(self):
            self.access_token = "access-value"
            self.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    storage = _FakeStorage()
    monkeypatch.setattr(creds, "Credentials", FakeCredentials)
    monkeypatch.setattr(creds, "get_storage_adapter", lambda: _async_value(storage))
    monkeypatch.setattr(
        creds, "get_user_email", lambda _credentials: _async_value("api@example.com")
    )
    monkeypatch.setattr(creds, "enable_required_apis", lambda *_args: _async_value(True))

    async def fake_fetch(**_kwargs):
        return "ag-project", "ultra"

    monkeypatch.setattr(creds, "fetch_project_id_and_tier", fake_fetch)
    monkeypatch.setattr(creds, "get_antigravity_api_url", lambda: _async_value("https://ag"))

    result = await creds.import_refresh_token_item(
        "refresh-value", email="provided@example.com", mode="antigravity"
    )

    assert result["status"] == "success"
    assert result["project_id"] == "ag-project"
    assert storage.stored[0][2] == "antigravity"
    assert storage.stored[0][1]["client_id"] == creds.ANTIGRAVITY_CLIENT_ID
    assert storage.stored[0][1]["client_secret"] == creds.ANTIGRAVITY_CLIENT_SECRET
    assert storage.stored[0][1]["scopes"] == creds.ANTIGRAVITY_SCOPES
    assert storage.stored[0][1]["token_uri"] == creds.TOKEN_URL
    assert storage.states[0][1]["user_email"] == "provided@example.com"
    assert storage.states[0][1]["tier"] == "ultra"


@pytest.mark.asyncio
async def test_geminicli_import_detects_project_and_enables_apis(monkeypatch):
    class FakeCredentials:
        def __init__(self, access_token, refresh_token, client_id, client_secret):
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

        async def refresh(self):
            self.access_token = "gemini-access"

    storage = _FakeStorage()
    enabled = []
    monkeypatch.setattr(creds, "Credentials", FakeCredentials)
    monkeypatch.setattr(creds, "get_storage_adapter", lambda: _async_value(storage))
    monkeypatch.setattr(
        creds,
        "get_user_projects",
        lambda _credentials: _async_value([{"projectId": "gemini-project"}]),
    )
    monkeypatch.setattr(
        creds,
        "enable_required_apis",
        lambda _credentials, project_id: _record_async(enabled, project_id),
    )
    monkeypatch.setattr(creds, "get_user_email", lambda _credentials: _async_value(None))

    result = await creds.import_refresh_token_item("gemini-refresh", mode="geminicli")

    assert result["status"] == "success"
    assert result["project_id"] == "gemini-project"
    assert enabled == ["gemini-project"]
    assert storage.stored[0][1]["client_id"] == creds.CLIENT_ID
    assert storage.stored[0][1]["client_secret"] == creds.CLIENT_SECRET
    assert storage.stored[0][1]["scopes"] == creds.SCOPES


@pytest.mark.asyncio
async def test_refresh_failure_does_not_store(monkeypatch):
    class FailedCredentials:
        def __init__(self, *args, **kwargs):
            self.refresh_token = kwargs.get("refresh_token")

        async def refresh(self):
            raise RuntimeError("refresh failed")

    storage = _FakeStorage()
    monkeypatch.setattr(creds, "Credentials", FailedCredentials)
    monkeypatch.setattr(creds, "get_storage_adapter", lambda: _async_value(storage))

    result = await creds.import_refresh_token_item("bad-refresh")

    assert result["status"] == "error"
    assert storage.stored == []
    assert storage.states == []


@pytest.mark.asyncio
async def test_complete_credential_upload_keeps_legacy_path(monkeypatch):
    calls = []

    async def fake_add(filename, data):
        calls.append((filename, data))

    monkeypatch.setattr(creds.credential_manager, "add_credential", fake_add)
    payload = {
        "client_id": "client",
        "client_secret": "secret",
        "token": "access",
        "refresh_token": "refresh",
        "scopes": ["scope"],
        "token_uri": creds.TOKEN_URL,
        "project_id": "project",
    }

    response = await creds.upload_credentials_common([_upload("credential.json", payload)])
    body = _response_body(response)

    assert body["uploaded_count"] == 1
    assert calls == [("credential.json", payload)]


def test_partial_credential_is_normalized_as_refresh_token_export():
    payload = {
        "client_id": "client",
        "access_token": "access",
        "refresh_token": "refresh",
    }

    assert creds._extract_refresh_token_records(payload) == [{"refresh_token": "refresh"}]


@pytest.mark.asyncio
async def test_all_refresh_token_failures_return_http_400(monkeypatch):
    async def fake_import(*_args, **_kwargs):
        return {
            "filename": "rt_failed.json",
            "status": "error",
            "message": "导入失败",
        }

    monkeypatch.setattr(creds, "import_refresh_token_item", fake_import)

    direct = await creds.import_refresh_token_records(
        [{"refresh_token": "failed-refresh"}], mode="antigravity"
    )
    uploaded = await creds.upload_credentials_common(
        [_upload("rt-export.json", [{"refresh_token": "failed-refresh"}])],
        mode="antigravity",
    )

    assert direct.status_code == 400
    assert uploaded.status_code == 400
    assert _response_body(direct)["imported_count"] == 0
    assert _response_body(uploaded)["uploaded_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_value", ["   ", 123, None])
async def test_invalid_refresh_token_shape_is_not_stored_as_legacy_credential(
    monkeypatch, invalid_value
):
    stored = []

    async def fake_add(*args):
        stored.append(args)

    monkeypatch.setattr(creds.credential_manager, "add_credential", fake_add)

    response = await creds.upload_credentials_common(
        [_upload("invalid-rt.json", {"refresh_token": invalid_value})]
    )

    assert response.status_code == 400
    assert _response_body(response)["uploaded_count"] == 0
    assert stored == []


def test_nested_refresh_tokens_are_all_extracted_from_same_object():
    payload = {
        "refresh_token": "outer",
        "nested": {"refreshToken": "inner"},
    }

    assert creds._extract_refresh_token_records(payload) == [
        {"refresh_token": "outer"},
        {"refresh_token": "inner"},
    ]


@pytest.mark.asyncio
async def test_mixed_container_preserves_complete_credential_and_imports_rt(monkeypatch):
    legacy_calls = []
    imported = []

    async def fake_add(filename, data):
        legacy_calls.append((filename, data))

    async def fake_import(refresh_token, **_kwargs):
        imported.append(refresh_token)
        return {"filename": "rt_imported.json", "status": "success", "message": "导入成功"}

    monkeypatch.setattr(creds.credential_manager, "add_credential", fake_add)
    monkeypatch.setattr(creds, "import_refresh_token_item", fake_import)
    complete = {
        "client_id": "client",
        "client_secret": "secret",
        "token": "access",
        "refresh_token": "complete-refresh",
        "project_id": "project",
    }
    payload = [complete, {"refresh_token": "rt-only"}]

    response = await creds.upload_credentials_common([_upload("mixed.json", payload)])
    body = _response_body(response)

    assert response.status_code == 200
    assert body["uploaded_count"] == 2
    assert len(legacy_calls) == 1
    assert legacy_calls[0][1] == complete
    assert imported == ["rt-only"]


def test_nested_credential_filename_distinguishes_account_and_project():
    oauth_base = {
        "client_id": "client",
        "client_secret": "secret",
        "token": "access",
        "refresh_token": "shared-refresh",
    }
    oauth_project_a = {**oauth_base, "project_id": "project-a"}
    oauth_project_b = {**oauth_base, "project_id": "project-b"}
    service_account_a = {
        "type": "service_account",
        "project_id": "shared-project",
        "client_email": "one@example.com",
        "private_key_id": "key-one",
        "private_key": "private-one",
    }
    service_account_b = {
        **service_account_a,
        "client_email": "two@example.com",
        "private_key_id": "key-two",
        "private_key": "private-two",
    }

    assert creds._nested_credential_filename(
        "mixed.json", oauth_project_a
    ) != creds._nested_credential_filename("mixed.json", oauth_project_b)
    assert creds._nested_credential_filename(
        "mixed.json", service_account_a
    ) != creds._nested_credential_filename("mixed.json", service_account_b)


async def _async_value(value):
    return value


async def _record_async(values, value):
    values.append(value)
    return True
