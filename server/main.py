"""Uvicorn entrypoint.

    python -m server.main          # dev, autoreload off
    uvicorn server.app:app --reload   # dev with reload
"""
from __future__ import annotations

import os

import uvicorn


def _port() -> int:
    """The port to bind.

    PORT is what every managed host (Render, Railway, Fly, Heroku) injects,
    and a service that ignores it never becomes reachable: the platform health-
    checks a port nothing is listening on and marks the deploy failed.
    VS_API_PORT stays the local override and wins when both are set.
    """
    for var in ("VS_API_PORT", "PORT"):
        raw = os.getenv(var, "").strip()
        if raw.isdigit():
            return int(raw)
    return 8000


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=_port(),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
