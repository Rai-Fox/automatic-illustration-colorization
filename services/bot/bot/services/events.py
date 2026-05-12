from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from services.api.app.infrastructure.jobs import RedisJobEventSubscriber


class JobEventListener:
    def __init__(self, redis_url: str) -> None:
        self._subscriber = RedisJobEventSubscriber(redis_url)

    async def listen(self) -> AsyncIterator[dict[str, Any]]:
        async for event in self._subscriber.listen():
            yield event
