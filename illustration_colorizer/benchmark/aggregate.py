from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from illustration_colorizer.benchmark.datasets import (
    BenchmarkSample,
    load_benchmark_dataset,
)
from illustration_colorizer.benchmark.runner import _resize_for_panel, _safe_path_name
from shared.images import to_rgb_uint8
from shared.omegaconf import to_plain_mapping
from shared.paths import resolve_from_root

_CAPTION_HEIGHT = 96
_CAPTION_FONT_SIZE = 44
_CAPTION_BACKGROUND = (22, 22, 22)
_CAPTION_FOREGROUND = (245, 245, 245)
_REFERENCE_MODE_ORDER = {
    "fixed_by_title": 0,
    "previous_output_by_title": 1,
}
_TECHNICAL_RUN_SUFFIXES = (
    "_cuda_images_all",
    "_cuda_images16",
)


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


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _fit_caption_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if max_width <= 0:
        return ""
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    if max_width < _text_size(draw, ".", font)[0]:
        return ""

    suffix = "..."
    for length in range(len(text), 0, -1):
        candidate = f"{text[:length]}{suffix}"
        if _text_size(draw, candidate, font)[0] <= max_width:
            return candidate
    return suffix if _text_size(draw, suffix, font)[0] <= max_width else ""


def _caption_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", _CAPTION_FONT_SIZE)
    except OSError:
        try:
            return ImageFont.load_default(size=_CAPTION_FONT_SIZE)
        except TypeError:
            return ImageFont.load_default()


def _add_caption(tile: np.ndarray, label: str) -> np.ndarray:
    tile_rgb = to_rgb_uint8(tile)
    height, width = tile_rgb.shape[:2]
    canvas = Image.new(
        "RGB",
        (width, height + _CAPTION_HEIGHT),
        color=_CAPTION_BACKGROUND,
    )
    canvas.paste(Image.fromarray(tile_rgb), (0, 0))

    draw = ImageDraw.Draw(canvas)
    font = _caption_font()
    caption = _fit_caption_text(draw, label, font, max(0, width - 8))
    text_width, text_height = _text_size(draw, caption, font)
    draw.text(
        ((width - text_width) // 2, height + (_CAPTION_HEIGHT - text_height) // 2),
        caption,
        fill=_CAPTION_FOREGROUND,
        font=font,
    )
    return np.asarray(canvas)


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


def _run_dirs(model_root: Path) -> list[Path]:
    return [
        path
        for path in model_root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ] if model_root.exists() else []


def _latest_run_dir(model_root: Path) -> Path | None:
    run_dirs = _run_dirs(model_root)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: (path / "manifest.json").stat().st_mtime)


def _entry_label(model_id: str, run_id: str | None, output_dir: Path) -> str:
    if run_id is None and (output_dir / "manifest.json").exists():
        return model_id
    run_label = _safe_path_name(run_id or output_dir.name)
    for suffix in _TECHNICAL_RUN_SUFFIXES:
        if run_label.endswith(suffix):
            run_label = run_label[: -len(suffix)]
            break
    if run_label == model_id or run_label.startswith(f"{model_id}_"):
        return run_label
    return f"{model_id}_{run_label}"


def _manifest_reference_mode(manifest: dict[str, dict[str, Any]]) -> str | None:
    for metadata in manifest.values():
        reference_mode = metadata.get("reference_mode")
        if reference_mode and reference_mode != "none":
            return str(reference_mode)
    return None


def _reference_mode_sort_key(reference_mode: str) -> tuple[int, str]:
    return (_REFERENCE_MODE_ORDER.get(reference_mode, 100), reference_mode)


def _resolve_output_entry(
    *,
    model_id: str,
    run_id: str | None,
    output_dir: Path,
) -> _GeneratedModelEntry:
    if not output_dir.exists():
        raise FileNotFoundError(f"Model output directory not found: {output_dir}")
    if not (output_dir / "manifest.json").exists():
        raise FileNotFoundError(f"Manifest not found: {output_dir / 'manifest.json'}")
    return _GeneratedModelEntry(
        label=_safe_path_name(_entry_label(model_id, run_id, output_dir)),
        model_id=model_id,
        run_id=run_id,
        output_dir=output_dir,
    )


def _resolve_reference_mode_entries(
    *,
    model_id: str,
    model_root: Path,
) -> list[_GeneratedModelEntry]:
    candidates: dict[str, tuple[int, float, Path]] = {}
    for run_dir in _run_dirs(model_root):
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reference_mode = _manifest_reference_mode(manifest)
        if reference_mode is None:
            continue
        score = (len(manifest), manifest_path.stat().st_mtime, run_dir)
        current = candidates.get(reference_mode)
        if current is None or score[:2] > current[:2]:
            candidates[reference_mode] = score

    if len(candidates) <= 1:
        return []

    return [
        _resolve_output_entry(
            model_id=model_id,
            run_id=run_dir.name,
            output_dir=run_dir,
        )
        for reference_mode, (_, _, run_dir) in sorted(
            candidates.items(),
            key=lambda item: _reference_mode_sort_key(item[0]),
        )
    ]


def _resolve_generated_entries(
    generated_root: Path,
    spec: str,
) -> list[_GeneratedModelEntry]:
    model_id, run_id = _split_model_spec(spec)
    model_root = generated_root / model_id
    if run_id:
        return [
            _resolve_output_entry(
                model_id=model_id,
                run_id=run_id,
                output_dir=model_root / _safe_path_name(run_id),
            )
        ]
    if (model_root / "manifest.json").exists():
        return [
            _resolve_output_entry(
                model_id=model_id,
                run_id=None,
                output_dir=model_root,
            )
        ]

    reference_entries = _resolve_reference_mode_entries(
        model_id=model_id,
        model_root=model_root,
    )
    if reference_entries:
        return reference_entries

    output_dir = _latest_run_dir(model_root) or model_root
    return [
        _resolve_output_entry(
            model_id=model_id,
            run_id=output_dir.name if output_dir != model_root else None,
            output_dir=output_dir,
        )
    ]


def _balanced_sample_ids(
    samples: list[BenchmarkSample],
    allowed_sample_ids: set[str],
    *,
    group_key: str,
    max_images: int | None,
    random_seed: int | None = None,
) -> list[str]:
    rng = random.Random(random_seed)
    grouped_sample_ids: dict[str, list[str]] = {}
    has_group_values = False
    for sample in samples:
        sample_id = str(sample.sample_id)
        if sample_id not in allowed_sample_ids:
            continue
        group_value = sample.metadata.get(group_key)
        if group_value is None:
            group_value = sample.metadata.get("reference_group_value")
        if group_value is None:
            group_value = "__ungrouped__"
        else:
            has_group_values = True
        grouped_sample_ids.setdefault(str(group_value), []).append(sample_id)

    if not has_group_values:
        sample_ids = [
            str(sample.sample_id)
            for sample in samples
            if str(sample.sample_id) in allowed_sample_ids
        ]
        rng.shuffle(sample_ids)
        if max_images is not None:
            return sample_ids[: max(0, int(max_images))]
        return sample_ids

    for sample_ids in grouped_sample_ids.values():
        rng.shuffle(sample_ids)
    group_values = list(grouped_sample_ids)
    rng.shuffle(group_values)

    selected: list[str] = []
    max_group_len = max(
        (len(sample_ids) for sample_ids in grouped_sample_ids.values()),
        default=0,
    )
    limit = max(0, int(max_images)) if max_images is not None else None
    for offset in range(max_group_len):
        for group_value in group_values:
            sample_ids = grouped_sample_ids[group_value]
            if offset >= len(sample_ids):
                continue
            selected.append(sample_ids[offset])
            if limit is not None and len(selected) >= limit:
                return selected
    return selected


def _reference_group_key(reference_config: dict[str, Any] | None) -> str:
    if reference_config is None:
        return "title"
    return str(to_plain_mapping(reference_config).get("group_key", "title"))


def aggregate_generated_panels(
    *,
    project_root: Path,
    models: list[str],
    benchmark_output_dir: str = "outputs/benchmark",
    generated_dir_name: str = "generated",
    output_dir_name: str = "comparisons",
    max_images: int | None = None,
    random_seed: int | None = None,
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

    entries = [
        entry
        for spec in models
        for entry in _resolve_generated_entries(generated_root, spec)
    ]
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    model_sample_sets: list[set[str]] = []
    for entry in entries:
        manifests[entry.label] = json.loads(
            (entry.output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        model_sample_sets.append(
            {path.stem for path in entry.output_dir.glob("*.png") if path.is_file()}
        )

    model_common_sample_ids = set.intersection(*model_sample_sets)

    if samples is None:
        if dataset_config is None:
            raise ValueError("dataset_config is required when samples are not passed.")
        aggregate_dataset_config = to_plain_mapping(dataset_config)
        aggregate_dataset_config["limit"] = None
        samples = load_benchmark_dataset(
            project_root=project_root,
            dataset_config=aggregate_dataset_config,
            reference_config=reference_config,
        )
    sample_by_id = {str(sample.sample_id): sample for sample in samples}

    common_sample_ids = _balanced_sample_ids(
        samples,
        model_common_sample_ids,
        group_key=_reference_group_key(reference_config),
        max_images=max_images,
        random_seed=random_seed,
    )

    output_root = benchmark_root / output_dir_name
    output_root.mkdir(parents=True, exist_ok=True)
    for existing_path in output_root.glob("*.png"):
        existing_path.unlink()

    for sample_id in common_sample_ids:
        sample = sample_by_id[sample_id]
        input_image = to_rgb_uint8(sample.input_image)
        height, width = input_image.shape[:2]
        comparative_tiles = [input_image]
        comparative_labels = ["input"]
        if sample.target_image is not None:
            comparative_tiles.append(
                _resize_for_panel(sample.target_image, width=width, height=height)
            )
            comparative_labels.append("target")

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
            comparative_labels.append(entry.label)

        captioned_tiles = [
            _add_caption(tile, label)
            for tile, label in zip(
                comparative_tiles,
                comparative_labels,
                strict=True,
            )
        ]
        comparative_panel = np.concatenate(captioned_tiles, axis=1)
        Image.fromarray(comparative_panel).save(output_root / f"{sample_id}.png")

    return {
        "output_dir": str(output_root),
        "sample_count": len(common_sample_ids),
        "sample_ids": common_sample_ids,
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
