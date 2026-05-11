from __future__ import annotations

import numpy as np

from illustration_colorizer.benchmark.image_utils import (
    numpy_images_to_torch_batch,
    resolve_torch_device,
)


def test_resolve_torch_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "illustration_colorizer.benchmark.image_utils.torch.cuda.is_available",
        lambda: False,
    )

    resolved = resolve_torch_device("cuda")

    assert resolved.type == "cpu"


def test_numpy_images_to_torch_batch_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "illustration_colorizer.benchmark.image_utils.torch.cuda.is_available",
        lambda: False,
    )
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    batch = numpy_images_to_torch_batch([image], device="cuda")

    assert batch.device.type == "cpu"
    assert tuple(batch.shape) == (1, 3, 4, 4)
