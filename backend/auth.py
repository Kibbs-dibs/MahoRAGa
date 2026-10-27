"""
Scoped API authentication for Project Aletheia (backend/auth.py).

Policy (network-exposed threat model):
  * READ endpoints stay open (health, query, GET corpus/graph/metrics/sessions).
  * WRITE / DESTRUCTIVE endpoints require a bearer token equal to
    ``settings.aletheia_api_key``:
        - /api/v1/ingest, /api/v1/corpus/wipe, /api/v1/corpus/retry-failed,
          /api/v1/watcher/sync, /api/v1/evaluate,
          POST /api/v1/sessions, DELETE /api/v1/sessions/{id}

Key-not-configured behaviour:
  * If no key is set AND the server is bound to a non-loopback interface
    (0.0.0.0 or a routable address), write routes FAIL CLOSED with HTTP 503 —
    a network-exposed box must not allow anonymous wipes.
  * If no key is set AND the bind is loopback, write routes are allowed, so
    local development stays frictionless.

Comparison uses ``secrets.compare_digest`` to avoid timing side-channels.
"""

from __future__ import annotations

import ipaddress
import secrets
from typing import Optional

from fastapi import Header, HTTPException, status

from core.config import settings


def _bind_is_loopback() -> bool:
    """True when the configured bind host is a loopback address."""
    host = (getattr(settings, "web_ui_host", "127.0.0.1") or "").strip()
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Hostnames other than localhost are treated as potentially exposed.
        return False


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    """Pull the token out of an ``Authorization: Bearer <token>`` header."""
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def require_write_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency guarding write/destructive routes.

    Raises 401 on a missing/invalid token when a key is configured, or 503 when
    no key is configured on a network-exposed bind (fail closed).
    """
    configured = settings.aletheia_api_key

    if not configured:
        if _bind_is_loopback():
            return  # local dev convenience
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Write endpoints are disabled: no ALETHEIA_API_KEY is configured "
                "while the server is network-exposed. Set ALETHEIA_API_KEY to "
                "enable ingest/wipe/watcher/evaluate operations."
            ),
        )

    token = _extract_bearer(authorization)
    if not token or not secrets.compare_digest(token, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key for this write operation.",
            headers={"WWW-Authenticate": "Bearer"},
        )
