from dataclasses import dataclass
from pathlib import Path

import mlflow

from illustration_colorizer.train.config import TrainConfig


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    metrics: dict[str, float]


def train_colorization_model(
    config: TrainConfig,
    dataset: object,
    output_dir: Path,
) -> TrainingArtifacts:
    model_path = output_dir / f"{config.name}.pt"
    metrics = {"loss": 0.0}

    mlflow.log_params(
        {
            "name": config.name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(model_path))

    return TrainingArtifacts(model_path=model_path, metrics=metrics)
