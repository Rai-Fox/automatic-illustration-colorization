from dataclasses import dataclass
from pathlib import Path

from shared.hydra import load_config


@dataclass(frozen=True)
class BotSettings:
    token: str
    api_url: str
    redis_url: str


def load_settings() -> BotSettings:
    project_root = Path(__file__).resolve().parents[3]
    config = load_config(project_root / "services" / "bot" / "conf")
    token = str(config.bot.token)
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    return BotSettings(
        token=token,
        api_url=str(config.bot.api_url),
        redis_url=str(config.bot.redis_url),
    )
