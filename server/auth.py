"""Bearer-token auth for the API.

Off unless configured. Set ``VS_API_KEYS`` (comma-separated static tokens)
and/or ``VS_JWT_SECRET`` (HS* signing secret, or the public key for RS*/ES*)
to turn it on. When on, every ``/api`` route depends on :func:`require_auth`
except the ones listed in :data:`PUBLIC_PATHS` — health, the OpenAPI docs, and
keyframe images (referenced by plain ``<img>`` tags that can't send headers).

``jwt`` decoding uses PyJWT if installed; static API keys need nothing.
"""

from __future__ import annotations

import logging

from fastapi import Header, HTTPException, Request

from server import config

logger = logging.getLogger("server.auth")

# Health, and the ways in. A sign-in route behind the sign-in would be a
# door that needs its own key.
PUBLIC_PATHS = ("/api/health", "/api/auth/methods", "/api/auth/otp/",
                "/api/auth/google/", "/api/billing/plan")


def _principal_from_jwt(token: str) -> dict | None:
    if not config.JWT_SECRET:
        return None
    try:
        import jwt  # type: ignore
    except ImportError:  # pragma: no cover - optional dep
        logger.warning("VS_JWT_SECRET set but PyJWT not installed; ignoring JWT path")
        return None
    try:
        claims = jwt.decode(
            token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM],
            audience=config.JWT_AUDIENCE or None,
            options={"verify_aud": bool(config.JWT_AUDIENCE)},
        )
        return {"kind": "jwt", "sub": claims.get("sub"), "claims": claims}
    except Exception as e:  # jwt.PyJWTError and friends
        raise HTTPException(401, f"invalid token: {e}") from e


def authenticate(authorization: str | None) -> dict:
    """Return a principal dict or raise 401. Assumes auth is enabled."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    token = authorization.split(" ", 1)[1].strip()

    if token in config.API_KEYS:
        return {"kind": "api_key", "sub": f"key:{token[:6]}…"}

    principal = _principal_from_jwt(token)
    if principal is not None:
        return principal

    raise HTTPException(401, "unrecognised token")


def _is_public(path: str) -> bool:
    if not path.startswith("/api/"):
        return True  # docs, openapi.json, root — not part of the data API
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return True
    return "/keyframes/" in path  # referenced by bare <img> tags


async def require_auth(
    request: Request, authorization: str | None = Header(default=None),
) -> dict:
    """FastAPI dependency. No-op passthrough when auth is disabled."""
    if not config.AUTH_ENABLED or _is_public(request.url.path):
        return {"kind": "public" if config.AUTH_ENABLED else "anonymous", "sub": None}
    return authenticate(authorization)
