"""Paper-inspection FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from labverse import __version__
from labverse.api.routes.paper import router as paper_router

_DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_LOCAL_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
    r"10\.\d+\.\d+\.\d+|"
    r"100\.\d+\.\d+\.\d+|"
    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|"
    r"192\.168\.\d+\.\d+"
    r"):(5173|4173)$"
)


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create the paper-bundle inspection API.

    The publication artifact exposes only offline bundle inspection routes. The
    optional ``config_path`` parameter is accepted for compatibility with older
    scripts, but it is intentionally ignored.
    """
    del config_path

    app = FastAPI(title="LabVerse Paper Artifact API", version=__version__)

    cors_origins_env = os.environ.get("LABVERSE_CORS_ORIGINS")
    cors_origins = (
        [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        if cors_origins_env
        else _DEFAULT_CORS_ORIGINS
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=_LOCAL_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    )

    app.include_router(paper_router, prefix="/api")

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


__all__ = ["create_app"]
