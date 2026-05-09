from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BenchmarkMetric(ABC):
    name: str
    requires_ground_truth = False

    @abstractmethod
    def compute(
        self,
        *,
        x_images: list[np.ndarray],
        y_images: list[np.ndarray],
        g_images: list[np.ndarray] | None = None,
    ) -> Any:
        pass
