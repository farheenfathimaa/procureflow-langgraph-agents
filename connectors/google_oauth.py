"""Google OAuth 2.0 Authorization Code flow with durable token refresh."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.memory import MemoryStore

logger = logging.getLogger(__name__)

OAUTH_STATE_TTL_SECONDS = 600


class OAuthError(RuntimeError):
    """Base OAuth connector error."""


class OAuthNotConfiguredError(OAuthError):
    """Raised when client credentials are missing."""


class OAuthNotAuthorizedError(OAuthError):
    """Raised when there is no usable token and the user must authorise."""


class OAuthTokenStore:
    """Local token storage with restrictive filesystem permissions.

    This keeps the demo dependency-free. The token file is gitignored and
    mounted on a Docker volume. Production deployments should encrypt the
    refresh token at rest (KMS or a secrets manager).
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def save(self, token_info: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.path.parent, 0o700)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(token_info), encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OAuthError("Stored Google token is unreadable; reconnect the account.") from exc
        if not isinstance(data, dict):
            raise OAuthError("Stored Google token has an invalid format.")
        return data

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)


class GoogleOAuthManager:
    """Authorize against Google and refresh tokens with the authorization code flow."""

    def __init__(self, settings: Settings, memory: MemoryStore) -> None:
        self.settings = settings
        self.memory = memory
        self.store = OAuthTokenStore(settings.google_token_path)

    @property
    def configured(self) -> bool:
        return self.settings.google_oauth_configured

    def _client_config(self) -> dict[str, Any]:
        if not self.configured:
            raise OAuthNotConfiguredError(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
            )
        return {
            "web": {
                "client_id": self.settings.google_client_id,
                "client_secret": self.settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [self.settings.google_redirect_uri],
            }
        }

    def authorization_url(self) -> str:
        """Build a one-time, CSRF-protected Google authorization URL."""

        try:
            from google_auth_oauthlib.flow import Flow
        except ImportError as exc:  # pragma: no cover
            raise OAuthError("google-auth-oauthlib is not installed") from exc

        state = secrets.token_urlsafe(32)
        self.memory.set_json(
            f"oauth_state:{state}",
            {"created_at": datetime.now(UTC).isoformat()},
            ttl_seconds=OAUTH_STATE_TTL_SECONDS,
        )
        flow = Flow.from_client_config(
            self._client_config(),
            scopes=self.settings.google_scopes_list,
            redirect_uri=self.settings.google_redirect_uri,
            state=state,
        )
        url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return url

    def exchange_code(self, code: str, state: str) -> None:
        """Validate OAuth state and exchange the authorization code for tokens."""

        if not state or not code:
            raise OAuthError("OAuth callback is missing the code or state parameter.")
        state_key = f"oauth_state:{state}"
        if not self.memory.get_json(state_key):
            raise OAuthError("OAuth state is missing or expired. Restart the connection flow.")
        # One-time use: remove before exchanging to prevent callback replay.
        self.memory.delete(state_key)

        try:
            from google_auth_oauthlib.flow import Flow

            flow = Flow.from_client_config(
                self._client_config(),
                scopes=self.settings.google_scopes_list,
                redirect_uri=self.settings.google_redirect_uri,
                state=state,
            )
            flow.fetch_token(code=code)
            credentials = flow.credentials
            if not credentials.refresh_token:
                logger.warning(
                    "Google did not return a refresh token; use prompt=consent and reconnect."
                )
            self.store.save(json.loads(credentials.to_json()))
        except Exception as exc:
            logger.exception("Google OAuth code exchange failed")
            raise OAuthError("Google authorization failed. Please retry the connection.") from exc

    def credentials(self):
        """Load, validate and refresh stored credentials when necessary."""

        try:
            from google.auth.exceptions import RefreshError
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
        except ImportError as exc:  # pragma: no cover
            raise OAuthError("google-auth is not installed") from exc

        token_info = self.store.load()
        if not token_info:
            raise OAuthNotAuthorizedError(
                "Google account is not connected. Open /auth/google to authorize."
            )
        try:
            credentials = Credentials.from_authorized_user_info(token_info)
        except ValueError as exc:
            raise OAuthError("Stored Google token is invalid; reconnect the account.") from exc

        if credentials.expired:
            if not credentials.refresh_token:
                raise OAuthNotAuthorizedError(
                    "Google token expired and no refresh token is available; reconnect the account."
                )
            try:
                credentials.refresh(Request())
                self.store.save(json.loads(credentials.to_json()))
                logger.info("Refreshed Google OAuth access token")
            except RefreshError as exc:
                self.store.delete()
                raise OAuthNotAuthorizedError(
                    "Google refresh token was rejected; reconnect the account."
                ) from exc
            except Exception as exc:
                logger.exception("Transient Google token refresh failure")
                raise OAuthError("Could not refresh Google credentials. Try again later.") from exc
        return credentials

    def status(self) -> dict[str, Any]:
        """Return non-secret connection metadata for the API."""

        if not self.configured:
            return {
                "configured": False,
                "authorized": False,
                "token_path": str(self.store.path),
                "message": "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable OAuth.",
            }
        try:
            token_info = self.store.load()
        except OAuthError as exc:
            return {
                "configured": True,
                "authorized": False,
                "token_path": str(self.store.path),
                "message": str(exc),
            }
        if not token_info:
            return {
                "configured": True,
                "authorized": False,
                "token_path": str(self.store.path),
                "message": "Google account not connected.",
            }
        return {
            "configured": True,
            "authorized": bool(token_info.get("refresh_token") or token_info.get("token")),
            "token_path": str(self.store.path),
            "expiry": token_info.get("expiry"),
            "scopes": token_info.get("scopes", []),
        }

    def disconnect(self) -> None:
        """Delete the stored token locally."""

        self.store.delete()
