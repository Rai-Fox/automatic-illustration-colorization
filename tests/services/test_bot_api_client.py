from __future__ import annotations

import asyncio
import base64
from typing import Any

from services.bot.bot.services.api_client import ColorizationApiClient


class _FakeResponse:
    def __init__(self, payload: Any = None, content: bytes = b"") -> None:
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append(("GET", url, kwargs))
        if url.endswith("/models"):
            return _FakeResponse([{"model_id": "passthrough", "enabled": True}])
        if url.endswith("/result"):
            return _FakeResponse(content=b"png")
        return _FakeResponse({"job_id": "job-1", "status": "succeeded"})

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.requests.append(("POST", url, kwargs))
        if url.endswith("/colorize"):
            return _FakeResponse(
                {"image_base64": base64.b64encode(b"png").decode("ascii")}
            )
        return _FakeResponse({"job_id": "job-1", "status": "queued"})


def test_bot_api_client_calls_job_endpoints() -> None:
    fake_client = _FakeAsyncClient()
    api_client = ColorizationApiClient("http://api", http_client=fake_client)

    models = asyncio.run(api_client.list_models())
    job = asyncio.run(
        api_client.create_colorization_job(
            b"image",
            model_id="passthrough",
            chat_id=12345,
            reference_bytes=b"ref",
            seed=7,
            options={"size": 576},
        )
    )
    status = asyncio.run(api_client.get_job_status("job-1"))
    result = asyncio.run(api_client.get_job_result("job-1"))

    assert api_client.base_url == "http://api"
    assert models[0]["model_id"] == "passthrough"
    assert job["job_id"] == "job-1"
    assert status["status"] == "succeeded"
    assert result == b"png"
    post_kwargs = fake_client.requests[1][2]
    assert post_kwargs["data"]["model_id"] == "passthrough"
    assert post_kwargs["data"]["chat_id"] == 12345
    assert post_kwargs["data"]["seed"] == 7
    assert post_kwargs["data"]["options"] == '{"size": 576}'
    assert "reference" in post_kwargs["files"]


def test_bot_api_client_sync_mvp_endpoint() -> None:
    fake_client = _FakeAsyncClient()
    api_client = ColorizationApiClient("http://api", http_client=fake_client)

    result = asyncio.run(
        api_client.colorize(
            b"image",
            model_id="passthrough",
        )
    )

    assert result == b"png"
