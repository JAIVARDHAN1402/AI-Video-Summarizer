"""Uvicorn entrypoint.

    python -m server.main          # dev, autoreload off
    uvicorn server.app:app --reload   # dev with reload
"""
from __future__ import annotations

import uvicorn

from server import config


def main() -> None:
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(__import__("os").getenv("VS_API_PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
