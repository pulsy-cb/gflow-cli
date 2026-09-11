"""Job management and sequential execution for gflow REST API server."""

from __future__ import annotations

import asyncio
import base64
import os
import time
import uuid
from pathlib import Path

import structlog

from gflow_cli import profile_store
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
from gflow_cli.data.queries import list_projects
from gflow_cli.paths import image_output_path
from gflow_cli.server.models import (
    BatchImageGenerateRequest,
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


def _safe_resolve_profile(req_profile: str | None) -> str:
    """Resolve profile name safely with fallback to default in test/headless environments."""
    settings = get_settings()
    candidate = req_profile or getattr(settings, "profile", None)
    if candidate:
        return candidate
    try:
        return profile_store.resolve_profile(None)
    except Exception:
        return "default"


class JobManager:
    """In-memory job repository with per-profile concurrency control."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobResponse] = {}
        self._profile_locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._active_clients: dict[str, FlowApiClient] = {}
        self._idle_timers: dict[str, asyncio.TimerHandle] = {}
        self._pending_counts: dict[str, int] = {}
        self._idle_timeout_seconds: float = float(
            os.environ.get("GFLOW_SERVER_IDLE_TIMEOUT", "5.0")
        )

    def get_profile_lock(self, profile: str) -> asyncio.Lock:
        if profile not in self._profile_locks:
            self._profile_locks[profile] = asyncio.Lock()
        return self._profile_locks[profile]

    def get_active_client(self, profile: str) -> FlowApiClient | None:
        return self._active_clients.get(profile)

    def get_pending_count(self, profile: str) -> int:
        return self._pending_counts.get(profile, 0)

    def cancel_idle_teardown(self, profile_name: str) -> None:
        timer = self._idle_timers.pop(profile_name, None)
        if timer:
            timer.cancel()

    def schedule_idle_teardown(self, profile_name: str, delay: float | None = None) -> None:
        self.cancel_idle_teardown(profile_name)
        timeout = self._idle_timeout_seconds if delay is None else delay
        try:
            loop = asyncio.get_running_loop()
            self._idle_timers[profile_name] = loop.call_later(
                timeout,
                lambda: asyncio.create_task(self.close_client(profile_name)),
            )
        except RuntimeError:
            pass

    def on_job_submitted(self, profile_name: str) -> None:
        self.cancel_idle_teardown(profile_name)
        self._pending_counts[profile_name] = self._pending_counts.get(profile_name, 0) + 1

    def on_job_finished(self, profile_name: str, immediate_close: bool = False) -> None:
        current = self._pending_counts.get(profile_name, 1)
        remaining = max(0, current - 1)
        self._pending_counts[profile_name] = remaining
        logger.info("server.queue.status", profile=profile_name, remaining=remaining)
        if remaining == 0:
            if immediate_close:
                self.schedule_idle_teardown(profile_name, delay=0.1)
            else:
                self.schedule_idle_teardown(profile_name, delay=self._idle_timeout_seconds)

    async def get_or_create_client(
        self, profile_name: str, profile_dir: Path, out_dir: Path
    ) -> FlowApiClient:
        self.cancel_idle_teardown(profile_name)

        client = self._active_clients.get(profile_name)
        if client is not None:
            try:
                _ = client.page
                return client
            except RuntimeError:
                self._active_clients.pop(profile_name, None)

        client = FlowApiClient(profile_dir=profile_dir, out_dir=out_dir)
        await client.__aenter__()
        self._active_clients[profile_name] = client
        return client

    async def close_client(self, profile_name: str) -> None:
        lock = self.get_profile_lock(profile_name)
        async with lock:
            if self._pending_counts.get(profile_name, 0) > 0:
                return
            self.cancel_idle_teardown(profile_name)
            client = self._active_clients.pop(profile_name, None)
            if client is not None:
                logger.info("server.browser.close", profile=profile_name)
                try:
                    await client.__aexit__(None, None, None)
                except Exception as exc:
                    logger.warning("server.browser.close_error", error=str(exc))

    async def close_all_clients(self) -> None:
        """Close all cached browser sessions across all profiles."""
        for profile_name in list(self._active_clients.keys()):
            await self.close_client(profile_name)

    def get_job(self, job_id: str) -> JobResponse | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[JobResponse]:
        items = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return items[:limit]

    async def submit_image_job(self, req: ImageGenerateRequest) -> JobResponse:
        profile_name = _safe_resolve_profile(req.profile)
        self.on_job_submitted(profile_name)

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
        profile_name = _safe_resolve_profile(req.profile)
        self.on_job_submitted(profile_name)

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

    async def submit_image_batch_job(self, req: BatchImageGenerateRequest) -> JobResponse:
        profile_name = _safe_resolve_profile(req.profile)
        self.on_job_submitted(profile_name)

        job_id = f"batch_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = JobResponse(
            job_id=job_id,
            status="pending",
            task_type="batch_image",
            total=len(req.prompts),
            completed=0,
            created_at=now,
            check_url=f"/v1/jobs/{job_id}",
        )
        self._jobs[job_id] = job

        if req.wait:
            await self._run_image_batch_job(job, req)
            return job

        asyncio.create_task(self._run_image_batch_job(job, req))
        return job

    async def _run_image_job(self, job: JobResponse, req: ImageGenerateRequest) -> None:
        settings = get_settings()
        profile_name = _safe_resolve_profile(req.profile)
        profile_dir = settings.profile_subdir(profile_name)
        out_dir = settings.output_dir
        upload_dir = out_dir / "uploads"

        try:
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

                lock = self.get_profile_lock(profile_name)
                async with lock:
                    job.status = "processing"
                    logger.info("server.image_job.started", job_id=job.job_id, profile=profile_name)
                    client = await self.get_or_create_client(profile_name, profile_dir, out_dir)
                    try:
                        project_id = req.project
                        if not project_id:
                            rows = list_projects(
                                db_path=settings.resolved_db_path(),
                                profile=profile_name,
                                limit=1,
                                offset=0,
                            )
                            if rows:
                                project_id = rows[0].project_id
                            else:
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
                    except Exception:
                        await self.close_client(profile_name)
                        raise
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                logger.error("server.image_job.failed", job_id=job.job_id, error=str(exc))
        finally:
            self.on_job_finished(profile_name, immediate_close=False)

    async def _run_image_batch_job(self, job: JobResponse, req: BatchImageGenerateRequest) -> None:
        settings = get_settings()
        profile_name = _safe_resolve_profile(req.profile)
        profile_dir = settings.profile_subdir(profile_name)
        out_dir = settings.output_dir

        try:
            try:
                aspect_str = map_size_to_aspect(req.size, default_aspect=req.aspect)
                image_aspect = ImageAspect.from_cli(aspect_str)
                image_model = ImageModel.from_cli(req.model)

                lock = self.get_profile_lock(profile_name)
                async with lock:
                    job.status = "processing"
                    logger.info(
                        "server.batch_job.started",
                        job_id=job.job_id,
                        profile=profile_name,
                        total=len(req.prompts),
                    )
                    client = await self.get_or_create_client(profile_name, profile_dir, out_dir)
                    try:
                        project_id = req.project
                        if not project_id:
                            rows = list_projects(
                                db_path=settings.resolved_db_path(),
                                profile=profile_name,
                                limit=1,
                                offset=0,
                            )
                            if rows:
                                project_id = rows[0].project_id
                            else:
                                proj = await client.create_project(title="gflow api image batch")
                                project_id = proj.project_id

                        data_items: list[MediaItem] = []
                        failed_errors: list[str] = []

                        for idx, prompt_text in enumerate(req.prompts, start=1):
                            logger.info(
                                "server.batch_job.prompt_start",
                                job_id=job.job_id,
                                index=idx,
                                total=len(req.prompts),
                                prompt=prompt_text[:60],
                            )
                            gen_req = GenerateImageRequest(
                                prompt=prompt_text,
                                model=image_model,
                                aspect=image_aspect,
                                ref_paths=(),
                                count=req.n,
                            )
                            try:
                                if req.n == 1:
                                    img = await client.generate_image(
                                        project_id=project_id, req=gen_req
                                    )
                                    images = [img]
                                else:
                                    images = await client.generate_images_batch(
                                        project_id=project_id, req=gen_req, count=req.n
                                    )

                                for i, im in enumerate(images, start=1):
                                    target = image_output_path(
                                        out_dir, job_id=im.media_name, index=i
                                    )
                                    saved = Path(await client.download_image(im, target))
                                    data_items.append(
                                        MediaItem(
                                            url=f"/v1/files/{saved.name}",
                                            local_path=str(saved),
                                            media_name=im.media_name,
                                            media_type="image",
                                        )
                                    )
                            except Exception as p_exc:
                                logger.warning(
                                    "server.batch_job.prompt_failed",
                                    job_id=job.job_id,
                                    index=idx,
                                    error=str(p_exc),
                                )
                                failed_errors.append(f"Prompt #{idx} failed: {p_exc}")
                                if not req.continue_on_error:
                                    raise

                            job.data = list(data_items)
                            job.completed = idx

                        if data_items:
                            job.status = "succeeded"
                            if failed_errors:
                                job.error = "; ".join(failed_errors)
                        else:
                            job.status = "failed"
                            job.error = "; ".join(failed_errors) or "All prompts failed"

                        job.completed_at = time.time()
                        logger.info(
                            "server.batch_job.succeeded",
                            job_id=job.job_id,
                            total=len(req.prompts),
                            generated=len(data_items),
                        )
                    except Exception:
                        await self.close_client(profile_name)
                        raise
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                logger.error("server.batch_job.failed", job_id=job.job_id, error=str(exc))
        finally:
            self.on_job_finished(profile_name, immediate_close=True)

    async def _run_video_job(self, job: JobResponse, req: VideoGenerateRequest) -> None:
        settings = get_settings()
        profile_name = _safe_resolve_profile(req.profile)
        profile_dir = settings.profile_subdir(profile_name)
        out_dir = settings.output_dir
        upload_dir = out_dir / "uploads"

        try:
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
                video_aspect = (
                    VideoAspect.from_cli(req.aspect) if req.aspect else VideoAspect.PORTRAIT
                )

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

                lock = self.get_profile_lock(profile_name)
                async with lock:
                    job.status = "processing"
                    logger.info("server.video_job.started", job_id=job.job_id, profile=profile_name)
                    client = await self.get_or_create_client(profile_name, profile_dir, out_dir)
                    try:
                        project_id = req.project
                        if not project_id:
                            rows = list_projects(
                                db_path=settings.resolved_db_path(),
                                profile=profile_name,
                                limit=1,
                                offset=0,
                            )
                            if rows:
                                project_id = rows[0].project_id
                            else:
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
                    except Exception:
                        await self.close_client(profile_name)
                        raise
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()
                logger.error("server.video_job.failed", job_id=job.job_id, error=str(exc))
        finally:
            self.on_job_finished(profile_name, immediate_close=False)


job_manager = JobManager()
