from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(config_dir: Path, overrides: Iterable[str] | None = None) -> DictConfig:
    overrides_list = list(overrides) if overrides else []
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=overrides_list)
