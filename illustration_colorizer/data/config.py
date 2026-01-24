from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path
    processed_dir: Path
    models_dir: Path


def default_data_paths(project_root: Path) -> DataPaths:
    data_dir = project_root / "data"
    return DataPaths(
        raw_dir=data_dir / "raw",
        processed_dir=data_dir / "processed",
        models_dir=data_dir / "models",
    )


def data_paths_from_config(config: DictConfig) -> DataPaths:
    return DataPaths(
        raw_dir=Path(config.data.raw_dir),
        processed_dir=Path(config.data.processed_dir),
        models_dir=Path(config.data.models_dir),
    )
