from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.api.app.container import AppContainer
from services.api.app.core.config import load_settings
from services.api.app.factory import create_app
from services.api.app.infrastructure.jobs import InMemoryJobStore, build_job_record
from services.worker.app import ColorizationWorker


def _png_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def api_client(tmp_path: Path) -> TestClient:
    settings = replace(
        load_settings(),
        model_id="passthrough",
        model_path="",
        device="cpu",
        service_storage_dir=str(tmp_path / "service"),
        enabled_models=(),
    )
    container = AppContainer.from_settings(settings, job_store=InMemoryJobStore())
    app = create_app(settings=settings, container=container)
    return TestClient(app)


def _worker_from_client(api_client: TestClient) -> ColorizationWorker:
    container = api_client.app.state.container
    return ColorizationWorker(
        job_service=container.job_service,
        colorization_service=container.colorization_service,
        file_storage=container.file_storage,
    )


def test_get_models_returns_registry_models(api_client: TestClient) -> None:
    response = api_client.get("/models")

    assert response.status_code == 200
    model_ids = {model["model_id"] for model in response.json()}
    assert "passthrough" in model_ids
    assert "colorcomic_auto" in model_ids


def test_sync_colorize_passthrough(api_client: TestClient) -> None:
    response = api_client.post(
        "/colorize",
        data={"model_id": "passthrough"},
        files={"file": ("image.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "passthrough"
    assert base64.b64decode(payload["image_base64"])
    assert payload["warnings"]


def test_reference_required_model_without_reference_returns_400(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/colorize",
        data={"model_id": "cgan_reference"},
        files={"file": ("image.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert "requires" in response.json()["detail"]


def test_job_lifecycle_create_process_result(api_client: TestClient) -> None:
    response = api_client.post(
        "/jobs",
        data={"model_id": "passthrough", "chat_id": "12345"},
        files={"file": ("image.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    queued = api_client.get(f"/jobs/{job_id}")
    assert queued.status_code == 200
    assert queued.json()["status"] == "queued"

    worker = _worker_from_client(api_client)
    assert asyncio.run(worker.run_once()) is True

    completed = api_client.get(f"/jobs/{job_id}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "succeeded"
    assert completed.json()["chat_id"] == 12345
    store = api_client.app.state.container.job_store
    assert store.events[-1]["status"] == "succeeded"
    assert store.events[-1]["chat_id"] == 12345
    result = api_client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.content.startswith(b"\x89PNG")


def test_failed_job_records_error(tmp_path: Path) -> None:
    settings = replace(
        load_settings(),
        model_id="passthrough",
        model_path="",
        device="cpu",
        service_storage_dir=str(tmp_path / "service"),
        enabled_models=(),
    )
    store = InMemoryJobStore()
    container = AppContainer.from_settings(settings, job_store=store)
    worker = ColorizationWorker(
        job_service=container.job_service,
        colorization_service=container.colorization_service,
        file_storage=container.file_storage,
    )
    job_id = "missing-input"
    record = build_job_record(
        job_id=job_id,
        model_id="passthrough",
        input_path=tmp_path / "missing.png",
        reference_paths=[],
        seed=None,
        options={},
    )
    asyncio.run(store.create(record))
    asyncio.run(store.enqueue(job_id))

    assert asyncio.run(worker.run_once()) is True

    failed = asyncio.run(store.get(job_id))
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"]
    assert store.events[-1]["status"] == "failed"
