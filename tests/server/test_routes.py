"""Tests for the gflow REST API server."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from gflow_cli.server.app import create_app
from gflow_cli.server.jobs import JobManager, save_base64_media
from gflow_cli.server.models import (
    map_size_to_aspect,
)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_root_endpoint(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "endpoints" in data
    assert data["endpoints"]["models"] == "/v1/models"


def test_health_endpoint(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_list_models(client: TestClient) -> None:
    res = client.get("/v1/models")
    assert res.status_code == 200
    catalog = res.json()
    assert "image" in catalog
    assert "video" in catalog
    # Check that our nano2-lite model is in the catalog
    image_aliases = [alias for m in catalog["image"]["models"] for alias in m["aliases"]]
    assert "nano2-lite" in image_aliases


@pytest.mark.asyncio
async def test_check_credits(client: TestClient) -> None:
    with patch(
        "gflow_cli.server.routes.inspect_profile",
        new=AsyncMock(return_value={"profile": "test", "credits": 88, "authenticated": True}),
    ):
        res = client.get("/v1/credits?profile=test")
        assert res.status_code == 200
        data = res.json()
        assert data["credits"] == 88


def test_map_size_to_aspect() -> None:
    assert map_size_to_aspect("1024x1024") == "1:1"
    assert map_size_to_aspect("1024x1792") == "9:16"
    assert map_size_to_aspect("1792x1024") == "16:9"
    assert map_size_to_aspect("1024x1365") == "3:4"
    assert map_size_to_aspect("1365x1024") == "4:3"
    assert map_size_to_aspect(None) == "9:16"


def test_save_base64_media(tmp_path: Path) -> None:
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    b64 = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"
    saved = save_base64_media(b64, tmp_path)
    assert saved.exists()
    assert saved.suffix == ".png"
    assert saved.read_bytes() == png_bytes


def test_image_generation_async(client: TestClient) -> None:
    payload = {
        "prompt": "a futuristic landscape",
        "model": "nano2-lite",
        "aspect": "16:9",
        "wait": False,
    }
    with patch.object(JobManager, "_run_image_job", new=AsyncMock()):
        res = client.post("/v1/images/generations", json=payload)
        assert res.status_code == 202
        data = res.json()
        assert data["job_id"].startswith("img_")
        assert data["task_type"] == "image"
        job_id = data["job_id"]

        # Poll job
        poll = client.get(f"/v1/jobs/{job_id}")
        assert poll.status_code == 200
        assert poll.json()["job_id"] == job_id


def test_video_generation_async(client: TestClient) -> None:
    payload = {
        "prompt": "a flying eagle over mountains",
        "model": "omni-flash",
        "resolution": "720p",
        "duration": 6,
        "wait": False,
    }
    with patch.object(JobManager, "_run_video_job", new=AsyncMock()):
        res = client.post("/v1/videos/generations", json=payload)
        assert res.status_code == 202
        data = res.json()
        assert data["job_id"].startswith("vid_")
        assert data["task_type"] == "video"


def test_get_job_not_found(client: TestClient) -> None:
    res = client.get("/v1/jobs/nonexistent_job_id")
    assert res.status_code == 404


def test_download_file_not_found(client: TestClient) -> None:
    res = client.get("/v1/files/nonexistent.png")
    assert res.status_code == 404


def test_image_batch_async(client: TestClient) -> None:
    payload = {
        "prompts": ["prompt 1", "prompt 2", "prompt 3"],
        "model": "nano2",
        "aspect": "9:16",
        "wait": False,
    }
    with patch.object(JobManager, "_run_image_batch_job", new=AsyncMock()):
        res = client.post("/v1/images/batches", json=payload)
        assert res.status_code == 202
        data = res.json()
        assert data["job_id"].startswith("batch_")
        assert data["task_type"] == "batch_image"
        assert data["total"] == 3
        assert data["completed"] == 0
        job_id = data["job_id"]

        poll = client.get(f"/v1/jobs/{job_id}")
        assert poll.status_code == 200
        assert poll.json()["job_id"] == job_id
        assert poll.json()["total"] == 3


def test_image_batch_sync(client: TestClient) -> None:
    payload = {
        "prompts": ["first prompt", "second prompt"],
        "model": "nano2",
        "aspect": "16:9",
        "wait": True,
    }
    with patch.object(JobManager, "_run_image_batch_job", new=AsyncMock()) as mock_run:
        res = client.post("/v1/images/batches", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["job_id"].startswith("batch_")
        assert data["task_type"] == "batch_image"
        assert data["total"] == 2
        mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_tracking_and_auto_close() -> None:
    mgr = JobManager()
    profile = "test_profile"

    assert mgr.get_pending_count(profile) == 0

    mgr.on_job_submitted(profile)
    assert mgr.get_pending_count(profile) == 1

    mgr.on_job_submitted(profile)
    assert mgr.get_pending_count(profile) == 2

    # First job finishes, 1 job remaining
    mgr.on_job_finished(profile, immediate_close=False)
    assert mgr.get_pending_count(profile) == 1
    assert profile not in mgr._idle_timers

    # Second job finishes, queue drained
    mgr.on_job_finished(profile, immediate_close=False)
    assert mgr.get_pending_count(profile) == 0
    assert profile in mgr._idle_timers

    # Cleanup timer
    mgr.cancel_idle_teardown(profile)
    assert profile not in mgr._idle_timers


def test_close_browser_endpoint(client: TestClient) -> None:
    res = client.post("/v1/browser/close")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
