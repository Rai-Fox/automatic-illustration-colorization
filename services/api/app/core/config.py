from dataclasses import dataclass
from pathlib import Path

from shared.hydra import load_component_config
from shared.paths import get_project_root


@dataclass(frozen=True)
class ApiSettings:
    host: str
    port: int
    log_level: str
    model_id: str
    model_path: str
    device: str
    max_image_size: int
    max_upload_bytes: int
    redis_url: str
    service_storage_dir: str
    job_ttl_seconds: int
    enabled_models: tuple[str, ...]


def load_settings() -> ApiSettings:
    project_root = get_project_root(Path(__file__), levels_up=4)
    config = load_component_config(project_root, "services/api/conf")
    enabled_models_raw = str(config.api.enabled_models).strip()
    enabled_models = tuple(
        model.strip() for model in enabled_models_raw.split(",") if model.strip()
    )
    return ApiSettings(
        host=str(config.api.host),
        port=int(config.api.port),
        log_level=str(config.api.log_level),
        model_id=str(config.api.model_id),
        model_path=str(config.api.model_path),
        device=str(config.api.device),
        max_image_size=int(config.api.max_image_size),
        max_upload_bytes=int(config.api.max_upload_bytes),
        redis_url=str(config.api.redis_url),
        service_storage_dir=str(config.api.service_storage_dir),
        job_ttl_seconds=int(config.api.job_ttl_seconds),
        enabled_models=enabled_models,
    )
