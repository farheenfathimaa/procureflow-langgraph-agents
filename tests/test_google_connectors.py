"""Google OAuth token handling and Google Sheets budget parsing (no network)."""

from __future__ import annotations

import json
import stat
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from app.config import Settings
from app.memory import MemoryStore
from connectors.google_oauth import (
    GoogleOAuthManager,
    OAuthError,
    OAuthNotAuthorizedError,
    OAuthTokenStore,
)
from connectors.google_sheets import (
    BudgetConnectorError,
    GoogleSheetsBudgetProvider,
    GoogleSheetsConnector,
    LocalBudgetProvider,
    build_budget_provider,
    parse_budget_rows,
    parse_money,
)


def _settings(tmp_path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "google_client_id": "client-id.apps.googleusercontent.com",
        "google_client_secret": "client-secret",
        "google_token_path": str(tmp_path / "google_token.json"),
        "redis_url": "",
        "database_url": "",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$120,000", 120_000),
        ("USD 120000", 120_000),
        ("$1,234.50", 1234.5),
        ("(1,500)", -1500),
        (7_500, 7500),
        (7.5, 7.5),
        (None, None),
        ("no amount here", None),
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw) == expected


def test_parse_budget_rows_uses_explicit_values():
    snapshot = parse_budget_rows(
        [
            ["Total budget", "$120,000"],
            ["Committed", "$42,000"],
            ["Available", "$78,000"],
        ],
        source="google_sheets:Budget!A1:C10",
    )
    assert snapshot.total_budget_usd == 120_000
    assert snapshot.committed_usd == 42_000
    assert snapshot.available_usd == 78_000
    assert snapshot.source == "google_sheets:Budget!A1:C10"


def test_parse_budget_rows_derives_available_and_ignores_headers():
    snapshot = parse_budget_rows(
        [
            ["Metric", "Value"],
            ["Annual Budget", "$100,000"],
            ["Spent", "$25,000"],
        ],
        source="test",
    )
    assert snapshot.available_usd == 75_000


def test_parse_budget_rows_derives_total_from_available():
    snapshot = parse_budget_rows(
        [["Available", "$60,000"], ["Committed", "$40,000"]],
        source="test",
    )
    assert snapshot.total_budget_usd == 100_000


def test_parse_budget_rows_treats_single_value_as_total():
    snapshot = parse_budget_rows([["$50,000"]], source="test")
    assert snapshot.total_budget_usd == 50_000
    assert snapshot.committed_usd == 0
    assert snapshot.available_usd == 50_000


def test_parse_budget_rows_rejects_unusable_sheets():
    with pytest.raises(BudgetConnectorError):
        parse_budget_rows([["nonsense", "data"]], source="test")
    with pytest.raises(BudgetConnectorError):
        parse_budget_rows([["Committed", "100"]], source="test")
    with pytest.raises(BudgetConnectorError):
        parse_budget_rows([["Total budget", "(100)"]], source="test")


def test_token_store_round_trip_and_permissions(tmp_path):
    store = OAuthTokenStore(str(tmp_path / "nested" / "token.json"))
    token = {"token": "access", "refresh_token": "refresh", "scopes": ["scope"]}
    store.save(token)

    assert store.load() == token
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    store.delete()
    assert store.load() is None


def test_oauth_status_when_unconfigured(tmp_path):
    settings = Settings(
        _env_file=None,
        google_client_id="",
        google_client_secret="",
        google_token_path=str(tmp_path / "token.json"),
        redis_url="",
    )
    manager = GoogleOAuthManager(settings, MemoryStore(""))
    status = manager.status()

    assert status["configured"] is False
    assert status["authorized"] is False
    with pytest.raises(OAuthError):
        manager.authorization_url()
    with pytest.raises(OAuthNotAuthorizedError):
        manager.credentials()


def test_oauth_requires_authorized_account(tmp_path):
    manager = GoogleOAuthManager(_settings(tmp_path), MemoryStore(""))
    status = manager.status()
    assert status["configured"] is True
    assert status["authorized"] is False
    with pytest.raises(OAuthNotAuthorizedError):
        manager.credentials()


def test_authorization_url_validates_csrf_state(tmp_path):
    manager = GoogleOAuthManager(_settings(tmp_path), MemoryStore(""))
    url = manager.authorization_url()
    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert parsed.netloc == "accounts.google.com"
    assert query["client_id"] == ["client-id.apps.googleusercontent.com"]
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    state = query["state"][0]
    assert manager.memory.get_json(f"oauth_state:{state}") is not None

    with pytest.raises(OAuthError):
        manager.exchange_code("code", "not-a-real-state")


def test_code_exchange_stores_refresh_token(tmp_path, monkeypatch):
    manager = GoogleOAuthManager(_settings(tmp_path), MemoryStore(""))
    manager.memory.set_json("oauth_state:good-state", {"created_at": "now"})

    class FakeCredentials:
        refresh_token = "refresh-token"

        def to_json(self):
            return json.dumps(
                {
                    "token": "access-token",
                    "refresh_token": self.refresh_token,
                    "scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
                }
            )

    class FakeFlow:
        @classmethod
        def from_client_config(cls, *args, **kwargs):
            return cls()

        def fetch_token(self, code):
            assert code == "auth-code"

        @property
        def credentials(self):
            return FakeCredentials()

    import google_auth_oauthlib.flow

    monkeypatch.setattr(google_auth_oauthlib.flow, "Flow", FakeFlow)
    manager.exchange_code("auth-code", "good-state")

    stored = manager.store.load()
    assert stored["refresh_token"] == "refresh-token"
    # The state is single-use.
    assert manager.memory.get_json("oauth_state:good-state") is None
    assert manager.status()["authorized"] is True


class _RowsRequest:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def execute(self, num_retries=0):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


class _ValuesResource:
    def __init__(self, request):
        self.request = request

    def get(self, **kwargs):
        assert kwargs["spreadsheetId"] == "sheet-123"
        return self.request


class _SpreadsheetsResource:
    def __init__(self, request):
        self.request = request

    def values(self):
        return _ValuesResource(self.request)


class _FakeService:
    def __init__(self, request):
        self.request = request

    def spreadsheets(self):
        return _SpreadsheetsResource(self.request)


def test_sheets_connector_requires_sheet_id(tmp_path):
    settings = _settings(tmp_path, google_sheet_id="")
    connector = GoogleSheetsConnector(settings, GoogleOAuthManager(settings, MemoryStore("")))

    with pytest.raises(BudgetConnectorError):
        connector.fetch_budget()


def test_sheets_connector_parses_values(tmp_path):
    settings = _settings(tmp_path, google_sheet_id="sheet-123")
    request = _RowsRequest(
        response={
            "values": [
                ["Total budget", "$120,000"],
                ["Committed", "$42,000"],
                ["Available", "$78,000"],
            ]
        }
    )
    connector = GoogleSheetsConnector(settings, GoogleOAuthManager(settings, MemoryStore("")))
    connector._build_service = lambda: _FakeService(request)

    snapshot = connector.fetch_budget()
    assert snapshot.total_budget_usd == 120_000
    assert snapshot.available_usd == 78_000
    assert request.calls == 1


def test_sheets_connector_does_not_retry_permission_errors(tmp_path):
    settings = _settings(tmp_path, google_sheet_id="sheet-123")

    class HttpError(Exception):
        resp = SimpleNamespace(status=403)

    request = _RowsRequest(error=HttpError("forbidden"))
    connector = GoogleSheetsConnector(settings, GoogleOAuthManager(settings, MemoryStore("")))
    connector._build_service = lambda: _FakeService(request)

    with pytest.raises(BudgetConnectorError, match="denied"):
        connector.fetch_budget()
    assert request.calls == 1


def test_sheets_connector_retries_transient_errors(tmp_path, monkeypatch):
    settings = _settings(tmp_path, google_sheet_id="sheet-123")
    monkeypatch.setattr("connectors.google_sheets.time.sleep", lambda seconds: None)

    class Transient(Exception):
        resp = SimpleNamespace(status=503)

    attempts = {"count": 0}

    class FlakyRequest:
        def execute(self, num_retries=0):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise Transient("temporary")
            return {"values": [["Total budget", "$10,000"]]}

    connector = GoogleSheetsConnector(
        settings, GoogleOAuthManager(settings, MemoryStore("")), max_retries=1
    )
    connector._build_service = lambda: _FakeService(FlakyRequest())

    snapshot = connector.fetch_budget()
    assert snapshot.total_budget_usd == 10_000
    assert attempts["count"] == 2


def test_budget_provider_factory_uses_local_demo_only_without_oauth(tmp_path):
    local_settings = Settings(
        _env_file=None,
        google_client_id="",
        google_client_secret="",
        google_sheet_id="",
        redis_url="",
        google_token_path=str(tmp_path / "token.json"),
    )
    provider, connector = build_budget_provider(
        local_settings, GoogleOAuthManager(local_settings, MemoryStore(""))
    )
    assert isinstance(provider, LocalBudgetProvider)
    assert connector is None
    assert provider.fetch_budget().source == "local_demo"

    sheets_settings = _settings(tmp_path, google_sheet_id="sheet-123")
    provider, connector = build_budget_provider(
        sheets_settings, GoogleOAuthManager(sheets_settings, MemoryStore(""))
    )
    assert isinstance(provider, GoogleSheetsBudgetProvider)
    assert isinstance(connector, GoogleSheetsConnector)
