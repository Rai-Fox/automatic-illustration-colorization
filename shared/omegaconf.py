from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_plain_mapping(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        data = OmegaConf.to_container(config, resolve=True)
        if not isinstance(data, dict):
            raise TypeError("Expected OmegaConf config to resolve to a mapping.")
        return data

    return dict(config)
