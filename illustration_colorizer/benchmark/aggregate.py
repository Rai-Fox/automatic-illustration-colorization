from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from illustration_colorizer.benchmark.datasets import (
    BenchmarkSample,
    load_benchmark_dataset,
)
from illustration_colorizer.benchmark.runner import _resize_for_panel, _safe_path_name
from shared.images import to_rgb_uint8
from shared.paths import resolve_from_root


def _split_panel(panel: np.ndarray, columns: int) -> list[np.ndarray]:
    if columns <= 0:
        raise ValueError("columns must be positive.")
    height, width = panel.shape[:2]
    if width % columns != 0:
        raise ValueError(
            f"Panel width {width} is not divisible by declared columns={columns}."
        )
    tile_width = width // columns
    return [
        panel[:, index * tile_width : (index + 1) * tile_width, :]
        for index in range(columns)
    ]


@dataclass(frozen=True)
class _GeneratedModelEntry:
    label: str
    model_id: str
    run_id: str | None
    output_dir: Path


def _split_model_spec(spec: str) -> tuple[str, str | None]:
    if ":" not in spec:
        return spec, None
    model_id, run_id = spec.split(":", 1)
    return model_id, run_id or None


def _latest_run_dir(model_root: Path) -> Path | None:
    run_dirs = [
        path
        for path in model_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ] if model_root.exists() else []
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: (path / "manifest.json").stat().st_mtime)


def _resolve_generated_entry(generated_root: Path, spec: str) -> _GeneratedModelEntry:
    model_id, run_id = _split_model_spec(spec)
    model_root = generated_root / model_id
    if run_id:
        output_dir = model_root / _safe_path_name(run_id)
        label = f"{model_id}_{run_id}"
    elif (model_root / "manifest.json").exists():
        output_dir = model_root
        label = model_id
    else:
        output_dir = _latest_run_dir(model_root) or model_root
        label = (
            f"{model_id}_{output_dir.name}"
            if output_dir != model_root
            else model_id
        )

    if not output_dir.exists():
        raise FileNotFoundError(f"Model output directory not found: {output_dir}")
    if not (output_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Manifest not found: {output_dir / 'manifest.json'}")
    return _GeneratedModelEntry(
        label=_safe_path_name(label),
        model_id=model_id,
        run_id=run_id,
        output_dir=output_dir,
    )


def aggregate_generated_panels(
    *,
    project_root: Path,
    models: list[str],
    benchmark_output_dir: str = "outputs/benchmark",
    generated_dir_name: str = "generated",
    output_dir_name: str = "comparisons",
    max_images: int | None = None,
    dataset_config: dict[str, Any] | None = None,
    reference_config: dict[str, Any] | None = None,
    samples: list[BenchmarkSample] | None = None,
) -> dict[str, object]:
    if not models:
        raise ValueError("At least one model must be provided.")

    benchmark_root = resolve_from_root(project_root, benchmark_output_dir)
    if benchmark_root is None or not benchmark_root.exists():
        raise FileNotFoundError(
            f"Benchmark output directory not found: {benchmark_output_dir}"
        )

    generated_root = benchmark_root / generated_dir_name
    if not generated_root.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_root}")

    if samples is None:
        if dataset_config is None:
            raise ValueError("dataset_config is required when samples are not passed.")
        samples = load_benchmark_dataset(
            project_root=project_root,
            dataset_config=dataset_config,
            reference_config=reference_config,
        )
    sample_by_id = {str(sample.sample_id): sample for sample in samples}

    entries = [_resolve_generated_entry(generated_root, spec) for spec in models]
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    sample_sets: list[set[str]] = [set(sample_by_id)]
    for entry in entries:
        manifests[entry.label] = json.loads(
            (entry.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        sample_sets.append(
            {path.stem for path in entry.output_dir.glob("*.png") if path.is_file()}
        )

    common_sample_ids = sorted(set.intersection(*sample_sets))
    if max_images is not None:
        common_sample_ids = common_sample_ids[: max(0, int(max_images))]

    output_root = benchmark_root / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    for existing_path in output_root.glob("*.png"):
        existing_path.unlink()

    for sample_id in common_sample_ids:
        sample = sample_by_id[sample_id]
        input_image = to_rgb_uint8(sample.input_image)
        height, width = input_image.shape[:2]
        comparative_tiles = [input_image]
        if sample.target_image is not None:
            comparative_tiles.append(
                _resize_for_panel(sample.target_image, width=width, height=height)
            )

        for entry in entries:
            result_path = entry.output_dir / f"{sample_id}.png"
            result_image = np.asarray(Image.open(result_path).convert("RGB"))
            metadata = manifests[entry.label].get(sample_id, {})
            if metadata.get("format") != "result":
                result_image = _split_panel(
                    result_image,
                    int(metadata.get("columns", 1)),
                )[-1]
            comparative_tiles.append(
                _resize_for_panel(result_image, width=width, height=height)
            )

        comparative_panel = np.concatenate(comparative_tiles, axis=1)
        Image.fromarray(comparative_panel).save(output_root / f"{sample_id}.png")

    return {
        "output_dir": str(output_root),
        "sample_count": len(common_sample_ids),
        "models": [
            {
                "label": entry.label,
                "model_id": entry.model_id,
                "run_id": entry.run_id or entry.output_dir.name,
                "output_dir": str(entry.output_dir),
            }
            for entry in entries
        ],
    }
