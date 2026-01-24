from pathlib import Path

import mlflow

from illustration_colorizer.train.config import TrainConfig
from illustration_colorizer.train.evaluate import evaluate_model
from illustration_colorizer.train.training import train_colorization_model


def run_training(project_root: Path, config: TrainConfig) -> dict[str, float]:
    dataset = None
    mlflow.set_experiment(config.name)

    with mlflow.start_run():
        artifacts = train_colorization_model(
            config=config,
            dataset=dataset,
            output_dir=project_root / "data" / "models",
        )
        metrics = evaluate_model(artifacts.model_path, dataset)
        mlflow.log_metrics(metrics)
        return metrics
