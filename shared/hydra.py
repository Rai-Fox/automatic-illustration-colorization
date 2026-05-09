from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(config_dir: Path, overrides: Iterable[str] | None = None) -> DictConfig:
    overrides_list = list(overrides) if overrides else []
    resolved_config_dir = config_dir.resolve()
    with initialize_config_dir(config_dir=str(resolved_config_dir), version_base=None):
        return compose(config_name="config", overrides=overrides_list)


def append_override(
    overrides: list[str],
    key: str,
    value: str | int | float | bool | None,
) -> None:
    if value is None:
        return

    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)

    overrides.append(f"{key}={rendered}")


def extend_overrides(overrides: list[str], raw_overrides: Iterable[str]) -> None:
    for override in raw_overrides:
        if override:
            overrides.append(override)


def load_component_config(
    project_root: Path,
    relative_conf_dir: str | Path,
    overrides: Iterable[str] | None = None,
) -> DictConfig:
    return load_config(project_root / relative_conf_dir, overrides)
