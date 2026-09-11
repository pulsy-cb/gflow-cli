"""Pydantic models for the gflow REST API server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def map_size_to_aspect(size: str | None, default_aspect: str = "9:16") -> str:
    """Map common OpenAI dimensions to Flow aspect ratio strings."""
    if not size:
        return default_aspect
    s = size.strip().lower()
    mapping: dict[str, str] = {
        "1024x1024": "1:1",
        "512x512": "1:1",
        "1024x1792": "9:16",
        "768x1344": "9:16",
        "720x1280": "9:16",
        "1792x1024": "16:9",
        "1344x768": "16:9",
        "1280x720": "16:9",
        "1024x1365": "3:4",
        "1365x1024": "4:3",
    }
    return mapping.get(s, default_aspect)


class ImageGenerateRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Prompt describing the image"
    )
    model: str = Field(
        default="nano2", description="Model alias: 'nano2', 'nano-pro', 'nano2-lite', 'image4'"
    )
    aspect: str = Field(
        default="9:16", description="Aspect ratio: '9:16', '16:9', '1:1', '4:3', '3:4'"
    )
    size: str | None = Field(
        default=None, description="OpenAI size compatibility (e.g. '1024x1024', '1024x1792')"
    )
    n: int = Field(default=1, ge=1, le=4, description="Number of images to generate (1-4)")
    image_base64: str | None = Field(
        default=None, description="Base64 encoded reference image for image-to-image"
    )
    ref_paths: list[str] | None = Field(
        default=None, description="Local file paths for reference images"
    )
    project: str | None = Field(default=None, description="Existing Flow project ID")
    profile: str | None = Field(default=None, description="Profile override")
    wait: bool = Field(
        default=True, description="Wait for completion (sync) or return job ID immediately (async)"
    )


class VideoGenerateRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=2000, description="Motion prompt describing the video"
    )
    model: str = Field(
        default="omni-flash",
        description="Video model alias: 'omni-flash', 'veo-lite', 'veo-fast', 'veo-quality'",
    )
    aspect: str = Field(default="9:16", description="Aspect ratio: '9:16' or '16:9'")
    duration: int = Field(
        default=6, description="Duration in seconds (4, 6, 8, 10 for omni-flash; 4, 6, 8 for veo)"
    )
    resolution: str | None = Field(
        default="720p", description="Resolution: '360p' or '720p' (supported on omni-flash)"
    )
    n: int = Field(default=1, ge=1, le=4, description="Number of videos to generate (1-4)")
    mode: str = Field(default="t2v", description="Mode: 't2v', 'i2v', or 'r2v'")
    image_base64: str | None = Field(
        default=None, description="Base64 encoded start frame for image-to-video"
    )
    initial_frame: str | None = Field(
        default=None, description="Local file path or Flow media UUID for start frame"
    )
    end_frame: str | None = Field(
        default=None, description="Local file path or Flow media UUID for end frame"
    )
    ref_paths: list[str] | None = Field(
        default=None, description="Local file paths for reference-to-video"
    )
    project: str | None = Field(default=None, description="Existing Flow project ID")
    profile: str | None = Field(default=None, description="Profile override")
    wait: bool = Field(
        default=False, description="Wait for completion (sync) or return job ID immediately (async)"
    )


class MediaItem(BaseModel):
    url: str = Field(..., description="URL endpoint to download or stream the generated file")
    local_path: str = Field(..., description="Absolute path of the generated file on disk")
    media_name: str | None = Field(default=None, description="Flow media ID if available")
    media_type: Literal["image", "video"] = Field(
        default="image", description="Type of media generated"
    )


def _empty_media_list() -> list[MediaItem]:
    return []


class JobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: Literal["pending", "processing", "succeeded", "failed"] = Field(
        ..., description="Job execution status"
    )
    task_type: Literal["image", "video"] = Field(..., description="Type of generation task")
    created_at: float = Field(..., description="Unix timestamp of job creation")
    completed_at: float | None = Field(default=None, description="Unix timestamp of job completion")
    data: list[MediaItem] = Field(
        default_factory=_empty_media_list, description="List of generated media items upon success"
    )
    error: str | None = Field(default=None, description="Error message if the job failed")
    check_url: str = Field(..., description="Status polling URL for this job")
