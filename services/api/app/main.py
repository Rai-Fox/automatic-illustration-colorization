import base64

from fastapi import FastAPI, File, HTTPException, UploadFile

from services.api.app.core.config import load_settings
from services.api.app.schemas import ColorizeResponse
from services.api.app.services.colorization import colorize_image

app = FastAPI(title="Illustration Colorization API")
settings = load_settings()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/colorize", response_model=ColorizeResponse)
async def colorize(file: UploadFile = File(...)) -> ColorizeResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    output_bytes = colorize_image(content, settings.model_path)
    output_b64 = base64.b64encode(output_bytes).decode("ascii")
    return ColorizeResponse(image_base64=output_b64, model=settings.model_path)
