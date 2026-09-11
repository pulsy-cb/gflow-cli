"""Job management and sequential execution for gflow REST API server."""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path

import structlog

from gflow_cli.api.client import FlowApiClient
from gflow_cli.api.image import Aspect as ImageAspect
from gflow_cli.api.image import GenerateImageRequest
from gflow_cli.api.image import Model as ImageModel
from gflow_cli.api.video import (
    Aspect as VideoAspect,
)
from gflow_cli.api.video import (
    GenerateVideoRequest,
    VideoModel,
)
from gflow_cli.api.video import (
    Mode as VideoMode,
)
from gflow_cli.config import get_settings
from gflow_cli.paths import image_output_path
from gflow_cli.server.models import (
    ImageGenerateRequest,
    JobResponse,
    MediaItem,
    VideoGenerateRequest,
    map_size_to_aspect,
)

logger = structlog.get_logger(__name__)


def save_base64_media(data_str: str, target_dir: Path) -> Path:
    """Decode base64 image/video string and save to target directory."""
    if "," in data_str:
        _, data_str = data_str.split(",", 1)
    raw = base64.b64decode(data_str)
    ext = ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        ext = ".jpg"
    elif raw.startswith(b"\x89PNG"):
        ext = ".png"
    elif raw.startswith(b"RIFF") and b"WEBP" in raw[:16]:
        ext = ".webp"
    elif raw.startswith(b"\x00\x00\x00") and b"ftyp" in raw[:16]:
        ext = ".mp4"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"upload_{uuid.uuid4().hex[:12]}{ext}"
    path = target_dir / filename
    path.write_bytes(raw)
    return path


class JobManager:
    """In-memory job repository with per-profile concurrency control."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobResponse] = {}
        self._profile_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _get_profile_lock(self, profile: str) -> asyncio.Lock:
        if profile not in self._profile_locks:
            self._profile_locks[profile] = asyncio.Lock()
        return self._profile_locks[profile]

    def get_job(self, job_id: str) -> JobResponse | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobResponse]:
        items = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return items[:limit]

    async def submit_image_job(self, req: ImageGenerateRequest) -> JobResponse:
        job_id = f"img_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = JobResponse(
            job_id=job_id,
            status="pending",
            task_type="image",
            created_at=now,
            check_url=f"/v1/jobs/{job_id}",
        )
        self._jobs[job_id] = job

        if req.wait:
            await self._run_image_job(job, req)
            return job

        asyncio.create_task(self._run_image_job(job, req))
        return job

    async def submit_video_job(self, req: VideoGenerateRequest) -> JobResponse:
        job_id = f"vid_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = JobResponse(
            job_id=job_id,
            status="pending",
            task_type="video",
            created_at=now,
            check_url=f"/v1/jobs/{job_id}",
        )
        self._jobs[job_id] = job

        if req.wait:
            await self._run_video_job(job, req)
            return job

        asyncio.create_task(self._run_video_job(job, req))
        return job

    async def _run_image_job(self, job: JobResponse, req: ImageGenerateRequest) -> None:
        settings = get_settings()
        profile_name = req.profile or settings.profile or "default"
        profile_dir = settings.profile_subdir(profile_name)
        out_dir = settings.output_dir
        upload_dir = out_dir / "uploads"

        try:
            ref_paths: list[Path] = []
            if req.ref_paths:
                ref_paths.extend(Path(p) for p in req.ref_paths)
            if req.image_base64:
                saved = save_base64_media(req.image_base64, upload_dir)
                ref_paths.append(saved)

            aspect_str = map_size_to_aspect(req.size, default_aspect=req.aspect)
            image_aspect = ImageAspect.from_cli(aspect_str)
            image_model = ImageModel.from_cli(req.model)

            gen_req = GenerateImageRequest(
                prompt=req.prompt,
                model=image_model,
                aspect=image_aspect,
                ref_paths=tuple(ref_paths),
                count=req.n,
            )

            lock = self._get_profile_lock(profile_name)
            async with lock:
                job.status = "processing"
                logger.info("server.image_job.started", job_id=job.job_id, profile=profile_name)
                async with FlowApiClient(
                    profile_dir=profile_dir,
                    out_dir=out_dir,
                ) as client:
                    project_id = req.project
                    if not project_id:
                        proj = await client.create_project(title="gflow api images")
                        project_id = proj.project_id

                    if req.n == 1:
                        img = await client.generate_image(project_id=project_id, req=gen_req)
                        images = [img]
                    else:
                        images = await client.generate_images_batch(
                            project_id=project_id, req=gen_req, count=req.n
                        )

                    data_items: list[MediaItem] = []
                    for i, im in enumerate(images, start=1):
                        target = image_output_path(out_dir, job_id=im.media_name, index=i)
                        saved = Path(await client.download_image(im, target))
                        data_items.append(
                            MediaItem(
                                url=f"/v1/files/{saved.name}",
                                local_path=str(saved),
                                media_name=im.media_name,
                                media_type="image",
                            )
                        )
                    job.data = data_items
                    job.status = "succeeded"
                    job.completed_at = time.time()
                    logger.info(
                        "server.image_job.succeeded",
                        job_id=job.job_id,
                        count=len(images),
                    )
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = time.time()
            logger.error("server.image_job.failed", job_id=job.job_id, error=str(exc))

    async def _run_video_job(self, job: JobResponse, req: VideoGenerateRequest) -> None:
        settings = get_settings()
        profile_name = req.profile or settings.profile or "default"
        profile_dir = settings.profile_subdir(profile_name)
        out_dir = settings.output_dir
        upload_dir = out_dir / "uploads"

        try:
            start_image_path: Path | None = None
            if req.initial_frame:
                p = Path(req.initial_frame)
                if p.exists():
                    start_image_path = p
            elif req.image_base64:
                start_image_path = save_base64_media(req.image_base64, upload_dir)

            video_mode = VideoMode(req.mode.lower()) if req.mode else VideoMode.T2V
            if start_image_path and video_mode == VideoMode.T2V:
                video_mode = VideoMode.I2V

            video_model = VideoModel.from_cli(req.model) if req.model else None
            video_aspect = VideoAspect.from_cli(req.aspect) if req.aspect else VideoAspect.PORTRAIT

            end_path = (
                Path(req.end_frame) if req.end_frame and Path(req.end_frame).exists() else None
            )

            ref_images = tuple(Path(p) for p in req.ref_paths) if req.ref_paths else ()

            gen_req = GenerateVideoRequest(
                prompt=req.prompt,
                mode=video_mode,
                aspect=video_aspect,
                model=video_model,
                duration=req.duration,
                resolution=req.resolution,
                count=req.n,
                start_image=start_image_path,
                end_image=end_path,
                reference_images=ref_images,
            )

            lock = self._get_profile_lock(profile_name)
            async with lock:
                job.status = "processing"
                logger.info("server.video_job.started", job_id=job.job_id, profile=profile_name)
                async with FlowApiClient(
                    profile_dir=profile_dir,
                    out_dir=out_dir,
                ) as client:
                    project_id = req.project
                    if not project_id:
                        proj = await client.create_project(title="gflow api videos")
                        project_id = proj.project_id

                    video = await client.generate_video(project_id=project_id, req=gen_req)

                    if not video.status.succeeded or video.local_path is None:
                        err_msg = video.status.error_message or "Video generation failed"
                        raise RuntimeError(err_msg)

                    job.data = [
                        MediaItem(
                            url=f"/v1/files/{video.local_path.name}",
                            local_path=str(video.local_path),
                            media_name=video.status.media_id,
                            media_type="video",
                        )
                    ]
                    job.status = "succeeded"
                    job.completed_at = time.time()
                    logger.info("server.video_job.succeeded", job_id=job.job_id)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = time.time()
            logger.error("server.video_job.failed", job_id=job.job_id, error=str(exc))


job_manager = JobManager()
