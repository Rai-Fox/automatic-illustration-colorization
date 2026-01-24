from pydantic import BaseModel


class ColorizeResponse(BaseModel):
    image_base64: str
    model: str
