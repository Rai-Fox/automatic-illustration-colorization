from dataclasses import dataclass
from pathlib import Path

from shared.hydra import load_component_config
from shared.paths import get_project_root


@dataclass(frozen=True)
class BotSettings:
    token: str
    api_url: str
    redis_url: str
    database_url: str
    max_reference_bytes: int
    max_reference_side: int


def load_settings() -> BotSettings:
    project_root = get_project_root(Path(__file__), levels_up=3)
    config = load_component_config(project_root, "services/bot/conf")
    token = str(config.bot.token)
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    return BotSettings(
        token=token,
        api_url=str(config.bot.api_url),
        redis_url=str(config.bot.redis_url),
        database_url=str(config.bot.database_url),
        max_reference_bytes=int(config.bot.max_reference_bytes),
        max_reference_side=int(config.bot.max_reference_side),
    )
