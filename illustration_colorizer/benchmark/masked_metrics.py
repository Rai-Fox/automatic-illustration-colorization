from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from illustration_colorizer.benchmark.image_utils import (
    delta_e_lab,
    get_region_colors_from_images,
    median_lab_color,
)


def safe_mean(values: list[float]) -> float | None:
    """Return the mean or None for an empty list."""
    if len(values) == 0:
        return None

    return float(np.mean(values))


def compute_cover_conditioned_character_color_error_single(
    cover_rgb: np.ndarray,
    y_images: list[np.ndarray],
    cover_masks: dict[str, dict[str, np.ndarray]],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    character_id: str,
    region: str,
) -> float:
    """Compute cover-conditioned character color error for one character-region pair."""
    if character_id not in cover_masks:
        raise ValueError(f"Character {character_id} not found in cover_masks.")

    if region not in cover_masks[character_id]:
        raise ValueError(f"Region {region} not found in cover_masks[{character_id}].")

    cover_mask = cover_masks[character_id][region]
    cover_color = median_lab_color(cover_rgb, cover_mask)

    errors = []

    for image_rgb, masks_i in zip(y_images, image_masks):
        if character_id not in masks_i:
            continue

        if region not in masks_i[character_id]:
            continue

        mask_i = masks_i[character_id][region]

        if mask_i.sum() == 0:
            continue

        image_color = median_lab_color(image_rgb, mask_i)
        error = delta_e_lab(image_color, cover_color)
        errors.append(error)

    if len(errors) == 0:
        raise ValueError(f"No valid image masks for character={character_id}, region={region}.")

    return float(np.mean(errors))


def compute_cover_conditioned_character_color_error(
    cover_rgb: np.ndarray,
    y_images: list[np.ndarray],
    cover_masks: dict[str, dict[str, np.ndarray]],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    characters: list[str],
    regions: list[str],
) -> dict[str, Any]:
    """Compute aggregated cover-conditioned character color error."""
    by_pair = {}
    by_character = {character_id: [] for character_id in characters}
    by_region = {region: [] for region in regions}
    values = []

    for character_id in characters:
        by_pair[character_id] = {}

        for region in regions:
            try:
                value = compute_cover_conditioned_character_color_error_single(
                    cover_rgb=cover_rgb,
                    y_images=y_images,
                    cover_masks=cover_masks,
                    image_masks=image_masks,
                    character_id=character_id,
                    region=region,
                )
            except ValueError:
                value = None

            by_pair[character_id][region] = value

            if value is not None:
                values.append(value)
                by_character[character_id].append(value)
                by_region[region].append(value)

    return {
        "overall": safe_mean(values),
        "by_character": {character_id: safe_mean(vals) for character_id, vals in by_character.items()},
        "by_region": {region: safe_mean(vals) for region, vals in by_region.items()},
        "by_pair": by_pair,
    }


def compute_character_color_consistency_single(
    y_images: list[np.ndarray],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    character_id: str,
    region: str,
) -> float:
    """Compute color consistency for one character-region pair across images."""
    colors = get_region_colors_from_images(
        y_images=y_images,
        image_masks=image_masks,
        character_id=character_id,
        region=region,
    )

    if len(colors) < 2:
        raise ValueError(f"Need at least 2 valid images for character={character_id}, region={region}.")

    colors_array = np.stack(colors, axis=0)
    reference_color = np.median(colors_array, axis=0)
    distances = [delta_e_lab(color_i, reference_color) for color_i in colors_array]

    return float(np.mean(distances))


def compute_character_color_consistency(
    y_images: list[np.ndarray],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    characters: list[str],
    regions: list[str],
) -> dict[str, Any]:
    """Compute aggregated character color consistency."""
    by_pair = {}
    by_character = {character_id: [] for character_id in characters}
    by_region = {region: [] for region in regions}
    values = []

    for character_id in characters:
        by_pair[character_id] = {}

        for region in regions:
            try:
                value = compute_character_color_consistency_single(
                    y_images=y_images,
                    image_masks=image_masks,
                    character_id=character_id,
                    region=region,
                )
            except ValueError:
                value = None

            by_pair[character_id][region] = value

            if value is not None:
                values.append(value)
                by_character[character_id].append(value)
                by_region[region].append(value)

    return {
        "overall": safe_mean(values),
        "by_character": {character_id: safe_mean(vals) for character_id, vals in by_character.items()},
        "by_region": {region: safe_mean(vals) for region, vals in by_region.items()},
        "by_pair": by_pair,
    }


def compute_color_identity_flip_rate_single(
    y_images: list[np.ndarray],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    character_id: str,
    region: str,
    color_flip_threshold: float = 15.0,
) -> float:
    """Compute color identity flip rate for one character-region pair."""
    colors = get_region_colors_from_images(
        y_images=y_images,
        image_masks=image_masks,
        character_id=character_id,
        region=region,
    )

    if len(colors) < 2:
        raise ValueError(f"Need at least 2 valid images for character={character_id}, region={region}.")

    pairs = list(combinations(range(len(colors)), 2))
    flips = 0

    for i, j in pairs:
        distance = delta_e_lab(colors[i], colors[j])

        if distance > color_flip_threshold:
            flips += 1

    return float(flips / len(pairs))


def compute_color_identity_flip_rate(
    y_images: list[np.ndarray],
    image_masks: list[dict[str, dict[str, np.ndarray]]],
    characters: list[str],
    regions: list[str],
    color_flip_threshold: float = 15.0,
) -> dict[str, Any]:
    """Compute aggregated color identity flip rate."""
    by_pair = {}
    by_character = {character_id: [] for character_id in characters}
    by_region = {region: [] for region in regions}
    values = []

    for character_id in characters:
        by_pair[character_id] = {}

        for region in regions:
            try:
                value = compute_color_identity_flip_rate_single(
                    y_images=y_images,
                    image_masks=image_masks,
                    character_id=character_id,
                    region=region,
                    color_flip_threshold=color_flip_threshold,
                )
            except ValueError:
                value = None

            by_pair[character_id][region] = value

            if value is not None:
                values.append(value)
                by_character[character_id].append(value)
                by_region[region].append(value)

    return {
        "overall": safe_mean(values),
        "by_character": {character_id: safe_mean(vals) for character_id, vals in by_character.items()},
        "by_region": {region: safe_mean(vals) for region, vals in by_region.items()},
        "by_pair": by_pair,
    }
