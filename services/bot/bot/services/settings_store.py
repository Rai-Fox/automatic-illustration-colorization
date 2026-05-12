from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_USER_MODEL_ID = "ddcolor"
LOGGER = logging.getLogger(__name__)


@dataclass
class UserSettings:
    chat_id: int
    model_id: str = DEFAULT_USER_MODEL_ID
    seed: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    reference_image: bytes | None = None


class UserSettingsStore(Protocol):
    async def initialize(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def get(self, chat_id: int) -> UserSettings:
        ...

    async def save(self, settings: UserSettings) -> None:
        ...


class PostgresUserSettingsStore:
    def __init__(
        self,
        database_url: str,
        *,
        connect_attempts: int = 10,
        connect_retry_seconds: float = 2.0,
    ) -> None:
        self.database_url = database_url
        self.connect_attempts = connect_attempts
        self.connect_retry_seconds = connect_retry_seconds
        self._pool: Any | None = None
        self._schema_initialized = False

    async def initialize(self) -> None:
        if self._pool is None:
            try:
                import asyncpg
            except ImportError as exc:
                raise RuntimeError(
                    "asyncpg is required for PostgreSQL user settings storage."
                ) from exc
            last_error: Exception | None = None
            for _ in range(self.connect_attempts):
                try:
                    self._pool = await asyncpg.create_pool(self.database_url)
                    break
                except (OSError, asyncpg.PostgresError) as exc:
                    last_error = exc
                    await asyncio.sleep(self.connect_retry_seconds)
            if self._pool is None:
                raise RuntimeError(
                    "Could not connect to PostgreSQL user settings storage."
                ) from last_error
            LOGGER.info("connected to PostgreSQL user settings storage")
        if self._schema_initialized:
            return
        async with self._pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS telegram_user_settings (
                    chat_id BIGINT PRIMARY KEY,
                    model_id TEXT NOT NULL DEFAULT '{DEFAULT_USER_MODEL_ID}',
                    seed INTEGER NULL,
                    options JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reference_image BYTEA NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE telegram_user_settings
                ALTER COLUMN model_id SET DEFAULT '{DEFAULT_USER_MODEL_ID}'
                """
            )
        self._schema_initialized = True
        LOGGER.info("initialized PostgreSQL user settings schema")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._schema_initialized = False

    async def get(self, chat_id: int) -> UserSettings:
        await self.initialize()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO telegram_user_settings (chat_id)
                    VALUES ($1)
                    ON CONFLICT (chat_id) DO NOTHING
                    """,
                    chat_id,
                )
                row = await connection.fetchrow(
                    """
                    SELECT chat_id, model_id, seed, options, reference_image
                    FROM telegram_user_settings
                    WHERE chat_id = $1
                    """,
                    chat_id,
                )
        if row is None:
            raise RuntimeError("Could not load user settings.")
        LOGGER.info("loaded user settings chat_id=%s", chat_id)
        return self._row_to_settings(row)

    async def save(self, settings: UserSettings) -> None:
        await self.initialize()
        async with self._pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO telegram_user_settings (
                    chat_id,
                    model_id,
                    seed,
                    options,
                    reference_image,
                    updated_at
                )
                VALUES ($1, $2, $3, $4::jsonb, $5, now())
                ON CONFLICT (chat_id) DO UPDATE
                SET model_id = EXCLUDED.model_id,
                    seed = EXCLUDED.seed,
                    options = EXCLUDED.options,
                    reference_image = EXCLUDED.reference_image,
                    updated_at = now()
                """,
                settings.chat_id,
                settings.model_id,
                settings.seed,
                json.dumps(settings.options),
                settings.reference_image,
            )
        LOGGER.info(
            "saved user settings chat_id=%s model_id=%s seed_set=%s "
            "reference_set=%s options_keys=%s",
            settings.chat_id,
            settings.model_id,
            settings.seed is not None,
            settings.reference_image is not None,
            sorted(settings.options),
        )

    def _row_to_settings(self, row: Any) -> UserSettings:
        raw_options = row["options"]
        if isinstance(raw_options, str):
            options = json.loads(raw_options)
        else:
            options = dict(raw_options or {})
        reference_image = row["reference_image"]
        if isinstance(reference_image, memoryview):
            reference_image = reference_image.tobytes()
        return UserSettings(
            chat_id=int(row["chat_id"]),
            model_id=str(row["model_id"]),
            seed=row["seed"],
            options=options,
            reference_image=reference_image,
        )


class InMemoryUserSettingsStore:
    def __init__(self) -> None:
        self.records: dict[int, UserSettings] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get(self, chat_id: int) -> UserSettings:
        LOGGER.info("loaded in-memory user settings chat_id=%s", chat_id)
        return self.records.setdefault(chat_id, UserSettings(chat_id=chat_id))

    async def save(self, settings: UserSettings) -> None:
        self.records[settings.chat_id] = UserSettings(
            chat_id=settings.chat_id,
            model_id=settings.model_id,
            seed=settings.seed,
            options=dict(settings.options),
            reference_image=settings.reference_image,
        )
        LOGGER.info(
            "saved in-memory user settings chat_id=%s model_id=%s",
            settings.chat_id,
            settings.model_id,
        )


def apply_setting(
    settings: UserSettings,
    *,
    param: str,
    value: Any,
    clear_values: set[str],
) -> None:
    if param in {"model", "model_id"}:
        settings.model_id = str(value)
    elif param == "seed":
        if str(value).lower() in clear_values:
            settings.seed = None
        else:
            settings.seed = int(value)
    else:
        if str(value).lower() in clear_values:
            settings.options.pop(param, None)
        else:
            settings.options[param] = value


def settings_from_mapping(data: Mapping[str, Any]) -> UserSettings:
    return UserSettings(
        chat_id=int(data["chat_id"]),
        model_id=str(data.get("model_id", DEFAULT_USER_MODEL_ID)),
        seed=data.get("seed"),
        options=dict(data.get("options") or {}),
        reference_image=data.get("reference_image"),
    )
