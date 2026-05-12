from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

JOB_QUEUE_KEY = "colorization:jobs"
JOB_KEY_PREFIX = "colorization:job:"
JOB_EVENTS_CHANNEL = "colorization:job-events"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_job_id() -> str:
    return uuid.uuid4().hex


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


def serialize_job(data: Mapping[str, Any]) -> dict[str, str]:
    serialized: dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            serialized[key] = ""
            continue
        if isinstance(value, (dict, list)):
            serialized[key] = json.dumps(value)
        else:
            serialized[key] = str(value)
    return serialized


def deserialize_job(data: Mapping[str | bytes, str | bytes]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    json_fields = {"reference_paths", "options", "warnings"}
    nullable_fields = {"started_at", "finished_at", "result_path", "error"}
    for raw_key, raw_value in data.items():
        key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else raw_key
        value = raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value
        if key in json_fields:
            default_value = {} if key == "options" else []
            decoded[key] = json.loads(value) if value else default_value
        elif key == "seed":
            decoded[key] = int(value) if value else None
        elif key == "chat_id":
            decoded[key] = int(value) if value else None
        elif key in nullable_fields:
            decoded[key] = value or None
        else:
            decoded[key] = value
    decoded.setdefault("reference_paths", [])
    decoded.setdefault("options", {})
    decoded.setdefault("warnings", [])
    return decoded


def build_job_record(
    *,
    job_id: str,
    model_id: str,
    input_path: Path,
    reference_paths: list[Path],
    seed: int | None,
    options: dict[str, Any],
    chat_id: int | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "queued",
        "model_id": model_id,
        "chat_id": chat_id,
        "created_at": utc_now(),
        "started_at": None,
        "finished_at": None,
        "input_path": str(input_path),
        "reference_paths": [str(path) for path in reference_paths],
        "result_path": None,
        "seed": seed,
        "options": options,
        "warnings": [],
        "error": None,
    }


class JobStore(Protocol):
    async def create(self, record: Mapping[str, Any]) -> None:
        ...

    async def enqueue(self, job_id: str) -> None:
        ...

    async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
        ...

    async def get(self, job_id: str) -> dict[str, Any] | None:
        ...

    async def update(self, job_id: str, updates: Mapping[str, Any]) -> None:
        ...

    async def publish_event(self, event: Mapping[str, Any]) -> None:
        ...


class RedisJobStore:
    def __init__(self, redis_url: str, *, ttl_seconds: int) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client: Any | None = None

    async def _redis(self) -> Any:
        if self._client is None:
            try:
                from redis import asyncio as redis_asyncio
            except ImportError as exc:
                raise RuntimeError(
                    "redis package is required for RedisJobStore. "
                    "Install the api or worker dependency group."
                ) from exc
            self._client = redis_asyncio.from_url(
                self.redis_url,
                decode_responses=False,
            )
        return self._client

    async def create(self, record: Mapping[str, Any]) -> None:
        redis = await self._redis()
        key = job_key(str(record["job_id"]))
        await redis.hset(key, mapping=serialize_job(record))
        if self.ttl_seconds > 0:
            await redis.expire(key, self.ttl_seconds)

    async def enqueue(self, job_id: str) -> None:
        redis = await self._redis()
        await redis.rpush(JOB_QUEUE_KEY, job_id)

    async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
        redis = await self._redis()
        item = await redis.blpop(JOB_QUEUE_KEY, timeout=timeout_seconds)
        if item is None:
            return None
        _, raw_job_id = item
        if isinstance(raw_job_id, bytes):
            return raw_job_id.decode("utf-8")
        return raw_job_id

    async def get(self, job_id: str) -> dict[str, Any] | None:
        redis = await self._redis()
        data = await redis.hgetall(job_key(job_id))
        if not data:
            return None
        return deserialize_job(data)

    async def update(self, job_id: str, updates: Mapping[str, Any]) -> None:
        redis = await self._redis()
        await redis.hset(job_key(job_id), mapping=serialize_job(updates))
        if self.ttl_seconds > 0:
            await redis.expire(job_key(job_id), self.ttl_seconds)

    async def publish_event(self, event: Mapping[str, Any]) -> None:
        redis = await self._redis()
        await redis.publish(JOB_EVENTS_CHANNEL, json.dumps(dict(event)))


class InMemoryJobStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.queue: list[str] = []
        self.events: list[dict[str, Any]] = []

    async def create(self, record: Mapping[str, Any]) -> None:
        self.records[str(record["job_id"])] = dict(record)

    async def enqueue(self, job_id: str) -> None:
        self.queue.append(job_id)

    async def dequeue(self, *, timeout_seconds: int = 5) -> str | None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while not self.queue:
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.01)
        return self.queue.pop(0)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        record = self.records.get(job_id)
        return dict(record) if record is not None else None

    async def update(self, job_id: str, updates: Mapping[str, Any]) -> None:
        self.records.setdefault(job_id, {}).update(dict(updates))

    async def publish_event(self, event: Mapping[str, Any]) -> None:
        self.events.append(dict(event))


class RedisJobEventSubscriber:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        try:
            from redis import asyncio as redis_asyncio
        except ImportError as exc:
            raise RuntimeError(
                "redis package is required to listen for job completion events."
            ) from exc

        redis = redis_asyncio.from_url(self.redis_url, decode_responses=True)
        pubsub = redis.pubsub()
        await pubsub.subscribe(JOB_EVENTS_CHANNEL)
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                yield json.loads(str(message["data"]))
        finally:
            await pubsub.unsubscribe(JOB_EVENTS_CHANNEL)
            await pubsub.close()
            await redis.close()
