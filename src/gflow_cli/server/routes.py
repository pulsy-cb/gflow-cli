"""REST API router for gflow endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from gflow_cli import __version__, profile_store
from gflow_cli.cli_models import build_catalog
from gflow_cli.config import get_settings
from gflow_cli.errors import GFlowError
from gflow_cli.server.jobs import job_manager
from gflow_cli.server.models import (
    ImageGenerateRequest,
    JobResponse,
    VideoGenerateRequest,
)
from gflow_cli.services.credits import inspect_profile

router = APIRouter()


@router.get("/", summary="Server Root & Discovery")
async def root() -> dict[str, Any]:
    """Return API metadata, available endpoints, and links to documentation."""
    return {
        "status": "ok",
        "service": "gflow-cli",
        "version": __version__,
        "docs_url": "/docs",
        "endpoints": {
            "models": "/v1/models",
            "credits": "/v1/credits",
            "image_generations": "/v1/images/generations",
            "video_generations": "/v1/videos/generations",
            "jobs": "/v1/jobs",
            "files": "/v1/files/{filename}",
        },
    }


@router.get("/health", summary="Health Check")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/v1/models", summary="List Available Models", tags=["Models"])
@router.get("/api/v1/models", include_in_schema=False)
async def list_models() -> dict[str, Any]:
    """List image and video models, aspect ratios, aliases, and reference limits."""
    return build_catalog()


@router.get("/v1/credits", summary="Check Account Credits", tags=["Account"])
@router.get("/api/v1/credits", include_in_schema=False)
async def check_credits(
    profile: str | None = Query(None, description="Profile to check (default: active profile)"),
) -> dict[str, Any]:
    """Inspect Veo credit balance and account status."""
    resolved_profile = profile_store.resolve_profile(profile)
    lock = job_manager.get_profile_lock(resolved_profile)
    async with lock:
        try:
            return await inspect_profile(resolved_profile)
        except GFlowError as exc:
            status_code = (
                exc.status
                if exc.status is not None and 400 <= exc.status < 600
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            detail_msg = f"{exc.title}: {exc.detail}"
            if exc.remediation_hint:
                detail_msg = f"{detail_msg} -> {exc.remediation_hint}"
            raise HTTPException(
                status_code=status_code,
                detail=detail_msg,
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to inspect credits: {exc}",
            ) from exc


@router.post(
    "/v1/images/generations",
    summary="Generate Images (T2I / I2I)",
    response_model=JobResponse,
    tags=["Generation"],
)
@router.post("/api/v1/images/generations", include_in_schema=False)
async def generate_image(req: ImageGenerateRequest, response: Response) -> JobResponse:
    """Generate images from text or reference images (Imagen & Nano Banana).

    Set `wait=true` to block until completed, or `wait=false` for an asynchronous job ID.
    """
    job = await job_manager.submit_image_job(req)
    if not req.wait and job.status in ("pending", "processing"):
        response.status_code = status.HTTP_202_ACCEPTED
    elif job.status == "failed":
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return job


@router.post(
    "/v1/videos/generations",
    summary="Generate Videos (T2V / I2V / R2V)",
    response_model=JobResponse,
    tags=["Generation"],
)
@router.post("/api/v1/videos/generations", include_in_schema=False)
async def generate_video(req: VideoGenerateRequest, response: Response) -> JobResponse:
    """Generate videos with Veo / Omni Flash.

    Supports `--resolution` ('360p' or '720p'), duration, aspects, and start-frame animation.
    Defaults to `wait=false` (202 Accepted with pollable job ID).
    """
    job = await job_manager.submit_video_job(req)
    if not req.wait and job.status in ("pending", "processing"):
        response.status_code = status.HTTP_202_ACCEPTED
    elif job.status == "failed":
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return job


@router.get(
    "/v1/jobs/{job_id}",
    summary="Get Job Status",
    response_model=JobResponse,
    tags=["Jobs"],
)
@router.get("/api/v1/jobs/{job_id}", include_in_schema=False)
async def get_job_status(job_id: str) -> JobResponse:
    """Poll the status and output of a generation job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return job


@router.get(
    "/v1/jobs",
    summary="List Recent Jobs",
    response_model=list[JobResponse],
    tags=["Jobs"],
)
@router.get("/api/v1/jobs", include_in_schema=False)
async def list_recent_jobs(
    limit: int = Query(50, ge=1, le=100, description="Max jobs to return"),
) -> list[JobResponse]:
    """Retrieve history of recently submitted jobs."""
    return job_manager.list_jobs(limit=limit)


@router.get("/v1/files/{filename}", summary="Download/Stream Generated Media", tags=["Media"])
@router.get("/download/{filename}", include_in_schema=False)
async def download_file(filename: str) -> FileResponse:
    """Download or stream a generated image or video file."""
    settings = get_settings()
    root = settings.output_dir

    # Search in common directories
    candidates = [
        root / filename,
        root / "images" / filename,
        root / "videos" / filename,
        root / "uploads" / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            media_type = "video/mp4" if c.suffix.lower() == ".mp4" else "image/png"
            return FileResponse(c, media_type=media_type, filename=c.name)

    # Recursive fallback in root (capped to prevent deep traversals)
    matches = list(root.glob(f"**/{filename}"))
    if matches and matches[0].is_file():
        match_path = matches[0]
        media_type = "video/mp4" if match_path.suffix.lower() == ".mp4" else "image/png"
        return FileResponse(match_path, media_type=media_type, filename=match_path.name)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File {filename} not found")
