import base64

import httpx


async def colorize_via_api(api_url: str, image_bytes: bytes) -> bytes:
    async with httpx.AsyncClient() as client:
        files = {"file": ("image.png", image_bytes, "image/png")}
        response = await client.post(f"{api_url}/colorize", files=files, timeout=60.0)
        response.raise_for_status()
        payload = response.json()
    return base64.b64decode(payload["image_base64"])
