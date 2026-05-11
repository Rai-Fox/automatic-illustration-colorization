from __future__ import annotations

from illustration_colorizer.models.base import (
    ColorizationModel,
    ColorizationRequest,
    ColorizationResult,
)
from illustration_colorizer.models.runtime import result


class PassthroughColorizationModel(ColorizationModel):
    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def colorize(self, request: ColorizationRequest) -> ColorizationResult:
        return result(
            image=request.input_image,
            model_id=self.model_id,
            warnings=["Using passthrough colorization model."],
        )
