from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shared.images import load_rgb_image
from shared.paths import find_file_by_stem, resolve_from_root

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LOGGER = logging.getLogger(__name__)


class LazyImageArray:
    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def _load(self) -> np.ndarray:
        return np.asarray(self._loader())

    def __array__(self, dtype: Any = None) -> np.ndarray:
        array = self._load()
        if dtype is not None:
            return np.asarray(array, dtype=dtype)
        return np.asarray(array)

    def __getitem__(self, key: Any) -> Any:
        return self._load()[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    input_image: np.ndarray
    target_image: np.ndarray | None = None
    reference_image: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkDataset:
    samples: list[BenchmarkSample]
    metadata: dict[str, Any] = field(default_factory=dict)


def _reference_mode(reference_config: dict[str, Any] | None) -> str:
    return str((reference_config or {}).get("mode", "none"))


def _reference_group_key(reference_config: dict[str, Any] | None) -> str:
    return str((reference_config or {}).get("group_key", "title"))


def _reference_sampling(reference_config: dict[str, Any] | None) -> str:
    return str((reference_config or {}).get("sampling", "balanced_titles"))


def _group_indices_by_value(values: list[Any]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        groups.setdefault(str(value), []).append(index)
    return groups


def _balanced_indices(
    groups: dict[str, list[int]],
    *,
    limit: int | None,
) -> list[int]:
    selected: list[int] = []
    max_group_len = max((len(indices) for indices in groups.values()), default=0)
    for offset in range(max_group_len):
        for title in sorted(groups):
            indices = groups[title]
            if offset >= len(indices):
                continue
            selected.append(indices[offset])
            if limit is not None and len(selected) >= limit:
                return selected
    return selected


def _lazy_hf_image(dataset: Any, index: int, column: str) -> LazyImageArray:
    def _load() -> np.ndarray:
        image = dataset[index].get(column)
        if image is None:
            raise ValueError(f"HF Arrow row {index} has no image column {column!r}.")
        return np.asarray(image.convert("RGB"))

    return LazyImageArray(_load)


def load_folder_dataset(
    *,
    project_root: Path,
    input_dir: str,
    target_dir: str | None = None,
    reference_dir: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    input_root = resolve_from_root(project_root, input_dir)
    if input_root is None or not input_root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    target_root = resolve_from_root(project_root, target_dir)
    reference_root = resolve_from_root(project_root, reference_dir)

    input_paths = sorted(
        path
        for path in input_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if limit is not None:
        input_paths = input_paths[:limit]

    samples: list[BenchmarkSample] = []
    for input_path in input_paths:
        target_image = None
        reference_image = None

        if target_root is not None:
            target_path = find_file_by_stem(
                target_root, input_path.stem, IMAGE_EXTENSIONS
            )
            if target_path is not None:
                target_image = load_rgb_image(target_path)

        if reference_root is not None:
            reference_path = find_file_by_stem(
                reference_root, input_path.stem, IMAGE_EXTENSIONS
            )
            if reference_path is not None:
                reference_image = load_rgb_image(reference_path)

        samples.append(
            BenchmarkSample(
                sample_id=input_path.stem,
                input_image=load_rgb_image(input_path),
                target_image=target_image,
                reference_image=reference_image,
                metadata={"input_path": str(input_path)},
            )
        )

    return samples


def load_hf_arrow_dataset(
    *,
    project_root: Path,
    dataset_dir: str,
    limit: int | None = None,
    reference_config: dict[str, Any] | None = None,
) -> list[BenchmarkSample]:
    return load_hf_arrow_benchmark_dataset(
        project_root=project_root,
        dataset_dir=dataset_dir,
        limit=limit,
        reference_config=reference_config,
    ).samples


def load_hf_arrow_benchmark_dataset(
    *,
    project_root: Path,
    dataset_dir: str,
    limit: int | None = None,
    reference_config: dict[str, Any] | None = None,
) -> BenchmarkDataset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for hf_arrow benchmark datasets."
        ) from exc

    dataset_root = resolve_from_root(project_root, dataset_dir)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(f"HF Arrow dataset directory not found: {dataset_dir}")

    arrow_files = sorted(dataset_root.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow shards found in {dataset_root}")

    dataset = load_dataset(
        "arrow",
        data_files={"train": [str(path) for path in arrow_files]},
    )["train"]

    mode = _reference_mode(reference_config)
    group_key = _reference_group_key(reference_config)
    sampling = _reference_sampling(reference_config)
    if mode not in {"none", "fixed_by_title", "previous_output_by_title"}:
        raise ValueError(f"Unsupported reference mode: {mode}")
    if mode != "none" and sampling != "balanced_titles":
        raise ValueError(f"Unsupported reference sampling: {sampling}")

    selected_indices: list[int]
    reference_by_index: dict[int, tuple[int, str]] = {}
    dataset_metadata: dict[str, Any] = {
        "reference_mode": mode,
        "reference_group_key": group_key,
        "reference_sampling": sampling,
        "title_count": 0,
        "excluded_seed_samples": 0,
    }

    if group_key not in dataset.column_names:
        if mode != "none":
            raise ValueError(
                f"Reference mode {mode} requires HF Arrow column {group_key!r}."
            )
        selected_indices = list(range(len(dataset)))
        if limit is not None:
            selected_indices = selected_indices[: min(limit, len(selected_indices))]
    else:
        titles = [str(value) for value in dataset[group_key]]
        grouped = _group_indices_by_value(titles)
        if mode == "none":
            selected_indices = (
                _balanced_indices(grouped, limit=limit)
                if limit is not None
                else list(range(len(dataset)))
            )
            selected_titles = {titles[index] for index in selected_indices}
            dataset_metadata["title_count"] = len(selected_titles)
        else:
            eligible_by_title: dict[str, list[int]] = {}
            seed_by_title: dict[str, int] = {}
            for title, indices in grouped.items():
                if len(indices) < 2:
                    continue
                seed_by_title[title] = indices[0]
                eligible_by_title[title] = indices[1:]

            selected_indices = _balanced_indices(eligible_by_title, limit=limit)
            selected_titles = {titles[index] for index in selected_indices}
            dataset_metadata["title_count"] = len(selected_titles)
            dataset_metadata["excluded_seed_samples"] = len(selected_titles)

            reference_source = "fixed_gt" if mode == "fixed_by_title" else "gt_seed"
            for index in selected_indices:
                title = titles[index]
                reference_by_index[index] = (
                    seed_by_title[title],
                    reference_source,
                )

    LOGGER.info(
        "Creating %d lazy HF Arrow benchmark samples",
        len(selected_indices),
    )
    title_values = dataset[group_key] if group_key in dataset.column_names else None
    tag_values = dataset["tags"] if "tags" in dataset.column_names else None
    samples: list[BenchmarkSample] = []
    for index in selected_indices:
        if len(samples) and len(samples) % 500 == 0:
            LOGGER.info(
                "Created %d/%d lazy benchmark samples",
                len(samples),
                len(selected_indices),
            )
        reference_image = None
        reference_sample_id = None
        reference_source = "none"
        if index in reference_by_index:
            reference_index, reference_source = reference_by_index[index]
            reference_image = _lazy_hf_image(dataset, reference_index, "color_image")
            reference_sample_id = str(reference_index)

        samples.append(
            BenchmarkSample(
                sample_id=str(index),
                input_image=_lazy_hf_image(dataset, index, "bw_image"),
                target_image=_lazy_hf_image(dataset, index, "color_image"),
                reference_image=reference_image,
                metadata={
                    "title": title_values[index] if title_values is not None else None,
                    "reference_group_value": (
                        title_values[index] if title_values is not None else None
                    ),
                    "tags": tag_values[index] if tag_values is not None else None,
                    "reference_mode": mode,
                    "reference_sample_id": reference_sample_id,
                    "reference_source": reference_source,
                },
            )
        )

    LOGGER.info("Created %d lazy benchmark samples", len(samples))
    return BenchmarkDataset(samples=samples, metadata=dataset_metadata)


def load_benchmark_dataset(
    *,
    project_root: Path,
    dataset_config: dict[str, Any],
    reference_config: dict[str, Any] | None = None,
) -> list[BenchmarkSample]:
    return load_benchmark_dataset_with_metadata(
        project_root=project_root,
        dataset_config=dataset_config,
        reference_config=reference_config,
    ).samples


def load_benchmark_dataset_with_metadata(
    *,
    project_root: Path,
    dataset_config: dict[str, Any],
    reference_config: dict[str, Any] | None = None,
) -> BenchmarkDataset:
    source = str(dataset_config.get("source", "hf_arrow"))
    limit = dataset_config.get("limit")
    resolved_limit = int(limit) if limit is not None else None
    mode = _reference_mode(reference_config)

    if source == "hf_arrow":
        return load_hf_arrow_benchmark_dataset(
            project_root=project_root,
            dataset_dir=str(dataset_config["hf_dataset_dir"]),
            limit=resolved_limit,
            reference_config=reference_config,
        )

    if source == "folders":
        if mode != "none":
            raise ValueError(
                f"Reference mode {mode} is only supported for hf_arrow datasets."
            )
        samples = load_folder_dataset(
            project_root=project_root,
            input_dir=str(dataset_config["input_dir"]),
            target_dir=dataset_config.get("target_dir"),
            reference_dir=dataset_config.get("reference_dir"),
            limit=resolved_limit,
        )
        for sample in samples:
            if sample.reference_image is not None:
                sample.metadata["reference_source"] = "folder"
        return BenchmarkDataset(
            samples=samples,
            metadata={
                "reference_mode": "none",
                "reference_group_key": _reference_group_key(reference_config),
                "reference_sampling": _reference_sampling(reference_config),
                "title_count": 0,
                "excluded_seed_samples": 0,
            },
        )

    raise ValueError(f"Unsupported dataset source: {source}")
