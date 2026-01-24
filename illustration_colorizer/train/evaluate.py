from pathlib import Path


def evaluate_model(model_path: Path, dataset: object) -> dict[str, float]:
    return {"psnr": 0.0, "ssim": 0.0}
