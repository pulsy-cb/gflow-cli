"""FastAPI app assembly for gflow REST API server."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gflow_cli import __version__
from gflow_cli.config import get_settings
from gflow_cli.server.jobs import job_manager
from gflow_cli.server.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield
    await job_manager.close_all_clients()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="gflow REST API",
        description=(
            "Autonomous REST API for Google Flow media generation (Imagen & Veo). "
            "Supports text-to-image, image-to-image, text-to-video, image-to-video, "
            "model selection, resolution controls, and background job tracking."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8006,
    default_profile: str | None = None,
) -> None:
    """Run the REST API server via uvicorn."""
    if default_profile:
        settings = get_settings()
        settings.profile = default_profile

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
