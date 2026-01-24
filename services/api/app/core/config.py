from dataclasses import dataclass
from pathlib import Path

from shared.hydra import load_config


@dataclass(frozen=True)
class ApiSettings:
    host: str
    port: int
    log_level: str
    model_path: str
    max_image_size: int
    redis_url: str


def load_settings() -> ApiSettings:
    project_root = Path(__file__).resolve().parents[4]
    config = load_config(project_root / "services" / "api" / "conf")
    return ApiSettings(
        host=str(config.api.host),
        port=int(config.api.port),
        log_level=str(config.api.log_level),
        model_path=str(config.api.model_path),
        max_image_size=int(config.api.max_image_size),
        redis_url=str(config.api.redis_url),
    )
