from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
