from pathlib import Path

from illustration_colorizer.data.config import DataPaths
from illustration_colorizer.data.io import ensure_directories, list_images
from illustration_colorizer.data.preprocess import preprocess_image


def run_pipeline(paths: DataPaths) -> list[Path]:
    ensure_directories([paths.raw_dir, paths.processed_dir, paths.models_dir])
    images = list_images(paths.raw_dir)
    return [preprocess_image(image_path, paths.processed_dir) for image_path in images]
