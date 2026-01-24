from pathlib import Path

import fire

from illustration_colorizer.data.config import data_paths_from_config
from illustration_colorizer.data.pipeline import run_pipeline
from illustration_colorizer.train.config import TrainConfig
from illustration_colorizer.train.run_training import run_training
from shared.hydra import load_config


class ColorizerCLI:
    """CLI entrypoint for data prep and training."""

    def data(
        self,
        raw_dir: str | None = None,
        processed_dir: str | None = None,
        models_dir: str | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        overrides = []
        if raw_dir:
            overrides.append(f"data.raw_dir={raw_dir}")
        if processed_dir:
            overrides.append(f"data.processed_dir={processed_dir}")
        if models_dir:
            overrides.append(f"data.models_dir={models_dir}")
        config = load_config(
            project_root / "illustration_colorizer" / "conf", overrides
        )
        paths = data_paths_from_config(config)
        run_pipeline(paths)

    def train(
        self,
        name: str | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent
        overrides = []
        if name:
            overrides.append(f"train.name={name}")
        if epochs is not None:
            overrides.append(f"train.epochs={epochs}")
        if batch_size is not None:
            overrides.append(f"train.batch_size={batch_size}")
        if learning_rate is not None:
            overrides.append(f"train.learning_rate={learning_rate}")
        config = load_config(
            project_root / "illustration_colorizer" / "conf", overrides
        )
        train_cfg = TrainConfig(
            name=config.train.name,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
            learning_rate=config.train.learning_rate,
        )
        run_training(project_root, train_cfg)


def main() -> None:
    fire.Fire(ColorizerCLI)


if __name__ == "__main__":
    main()
