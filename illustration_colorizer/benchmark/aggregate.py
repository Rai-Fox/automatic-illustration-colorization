from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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


def aggregate_generated_panels(
    *,
    project_root: Path,
    models: list[str],
    benchmark_output_dir: str = "outputs/benchmark",
    generated_dir_name: str = "generated",
    output_dir_name: str = "comparisons",
    max_images: int | None = None,
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

    model_dirs = {model_id: generated_root / model_id for model_id in models}
    for _model_id, model_dir in model_dirs.items():
        if not model_dir.exists():
            raise FileNotFoundError(f"Model output directory not found: {model_dir}")

    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    sample_sets: list[set[str]] = []
    for model_id, model_dir in model_dirs.items():
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifests[model_id] = json.loads(manifest_path.read_text(encoding="utf-8"))
        sample_sets.append(
            {
                path.stem
                for path in model_dir.glob("*.png")
                if path.is_file()
            }
        )

    common_sample_ids = sorted(set.intersection(*sample_sets))
    if max_images is not None:
        common_sample_ids = common_sample_ids[: max(0, int(max_images))]

    output_root = benchmark_root / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    for existing_path in output_root.glob("*.png"):
        existing_path.unlink()

    for sample_id in common_sample_ids:
        first_model = models[0]
        first_panel = np.asarray(
            Image.open(model_dirs[first_model] / f"{sample_id}.png").convert("RGB")
        )
        first_metadata = manifests[first_model][sample_id]
        first_tiles = _split_panel(first_panel, int(first_metadata["columns"]))
        comparative_tiles = [first_tiles[0]]
        if bool(first_metadata["has_ground_truth"]):
            comparative_tiles.append(first_tiles[1])

        for model_id in models:
            panel = np.asarray(
                Image.open(model_dirs[model_id] / f"{sample_id}.png").convert("RGB")
            )
            metadata = manifests[model_id][sample_id]
            tiles = _split_panel(panel, int(metadata["columns"]))
            comparative_tiles.append(to_rgb_uint8(tiles[-1]))

        comparative_panel = np.concatenate(comparative_tiles, axis=1)
        Image.fromarray(comparative_panel).save(output_root / f"{sample_id}.png")

    return {
        "output_dir": str(output_root),
        "sample_count": len(common_sample_ids),
        "models": models,
    }
