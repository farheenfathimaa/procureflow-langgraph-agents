"""Google Sheets budget reader with OAuth refresh and retry handling."""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.config import Settings
from app.schemas import utc_now
from connectors.google_oauth import GoogleOAuthManager, OAuthError

logger = logging.getLogger(__name__)

_MONEY_RE = re.compile(r"-?\(?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\)?")
_KEY_ALIASES: dict[str, set[str]] = {
    "total": {
        "annualbudget",
        "budget",
        "budgetusd",
        "total",
        "totalbudget",
        "totalbudgetusd",
        "yearlybudget",
    },
    "committed": {
        "committed",
        "committedusd",
        "expenses",
        "spent",
        "spentusd",
        "used",
    },
    "available": {
        "available",
        "availableusd",
        "balance",
        "remaining",
        "remainingbudget",
    },
}


class BudgetConnectorError(RuntimeError):
    """Budget data could not be retrieved safely."""


class BudgetSnapshot(BaseModel):
    """Normalised budget values in USD."""

    total_budget_usd: float = Field(ge=0)
    committed_usd: float = Field(ge=0)
    available_usd: float = Field(ge=0)
    source: str
    fetched_at: str = Field(default_factory=utc_now)


class BudgetProvider(Protocol):
    """Interface consumed by the BudgetAgent."""

    def fetch_budget(self) -> BudgetSnapshot: ...


def parse_money(value: Any) -> float | None:
    """Parse ``$120,000``, ``USD 120000``, ``(1,500)`` and plain numbers."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    match = _MONEY_RE.search(text)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    amount = float(raw)
    if text.strip().startswith("(") and text.strip().endswith(")"):
        amount = -amount
    return amount


def parse_budget_rows(rows: list[list[Any]], source: str) -> BudgetSnapshot:
    """Parse a simple key/value budget sheet.

    Accepted layout (header optional):

    ``Total budget | 120000``
    ``Committed | 42000``
    ``Available | 78000``

    A single value cell is treated as the total budget.
    """

    found: dict[str, float] = {}
    for row in rows:
        if not row:
            continue
        if len(row) == 1:
            amount = parse_money(row[0])
            if amount is not None and "total" not in found:
                found["total"] = amount
            continue
        key = re.sub(r"[^a-z0-9]", "", str(row[0]).lower())
        amount = parse_money(row[1])
        if amount is None:
            continue
        for canonical, aliases in _KEY_ALIASES.items():
            if key in aliases and canonical not in found:
                found[canonical] = amount

    if not found:
        raise BudgetConnectorError(
            "The Google Sheet did not contain recognisable budget rows. Expected "
            "Total budget / Committed / Available in the first two columns."
        )

    total = found.get("total")
    committed = found.get("committed", 0.0)
    available = found.get("available")
    if total is None and available is not None:
        total = available + committed
    if total is None:
        raise BudgetConnectorError("The budget sheet must include a total budget value.")
    if available is None:
        available = total - committed
    if committed < 0 or total < 0:
        raise BudgetConnectorError(
            "The budget sheet contains a negative total or committed amount."
        )
    return BudgetSnapshot(
        total_budget_usd=max(total, 0.0),
        committed_usd=max(committed, 0.0),
        available_usd=max(available, 0.0),
        source=source,
    )


class LocalBudgetProvider:
    """Deterministic demo budget used when Google Sheets is not configured."""

    def __init__(self, total: float, committed: float) -> None:
        self.total = total
        self.committed = committed

    def fetch_budget(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            total_budget_usd=self.total,
            committed_usd=self.committed,
            available_usd=max(self.total - self.committed, 0.0),
            source="local_demo",
            fetched_at=datetime.now(UTC).isoformat(),
        )


class GoogleSheetsConnector:
    """Read a budget range from Google Sheets using OAuth 2.0 credentials."""

    def __init__(
        self,
        settings: Settings,
        oauth: GoogleOAuthManager,
        max_retries: int = 2,
    ) -> None:
        self.settings = settings
        self.oauth = oauth
        self.max_retries = max_retries
        self._service = None

    def _build_service(self):
        if self._service is not None:
            return self._service
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover
            raise BudgetConnectorError("google-api-python-client is not installed") from exc
        credentials = self.oauth.credentials()
        self._service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        return self._service

    def fetch_budget(self) -> BudgetSnapshot:
        """Fetch and parse the configured budget range, retrying transient errors."""

        if not self.settings.google_sheet_id:
            raise BudgetConnectorError(
                "GOOGLE_SHEET_ID is not configured; using local demo budget is disabled "
                "when OAuth is configured."
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                service = self._build_service()
                values = (
                    service.spreadsheets()
                    .values()
                    .get(
                        spreadsheetId=self.settings.google_sheet_id,
                        range=self.settings.google_sheet_range,
                    )
                    .execute(num_retries=0)
                )
                rows = values.get("values", [])
                source = f"google_sheets:{self.settings.google_sheet_range}"
                return parse_budget_rows(rows, source=source)
            except OAuthError:
                # Reauthorization cannot be fixed by retrying. Clear the cached
                # service so a later call uses refreshed credentials.
                self._service = None
                raise
            except Exception as exc:
                last_error = exc
                status = _http_status(exc)
                logger.warning(
                    "Google Sheets read failed (attempt %s/%s, status=%s)",
                    attempt + 1,
                    self.max_retries + 1,
                    status,
                )
                self._service = None
                if status in {401, 403}:
                    raise BudgetConnectorError(
                        "Google Sheets access was denied. Check the OAuth scopes and sheet sharing."
                    ) from exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4) + random.uniform(0, 0.2))
        raise BudgetConnectorError(
            "Google Sheets budget could not be read after retries. Approval is required."
        ) from last_error


def _http_status(exc: Exception) -> int | None:
    response = getattr(exc, "resp", None)
    return getattr(response, "status", None)


class GoogleSheetsBudgetProvider:
    """Budget provider that deliberately never silently falls back on errors."""

    def __init__(self, connector: GoogleSheetsConnector) -> None:
        self.connector = connector

    def fetch_budget(self) -> BudgetSnapshot:
        return self.connector.fetch_budget()


def build_budget_provider(
    settings: Settings, oauth: GoogleOAuthManager
) -> tuple[BudgetProvider, GoogleSheetsConnector | None]:
    """Choose Google Sheets when configured, otherwise the labelled demo budget."""

    if settings.google_sheets_configured:
        connector = GoogleSheetsConnector(settings, oauth)
        return GoogleSheetsBudgetProvider(connector), connector
    return (
        LocalBudgetProvider(settings.local_total_budget_usd, settings.local_committed_usd),
        None,
    )
