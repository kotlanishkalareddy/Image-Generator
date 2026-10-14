import os
import requests
from io import BytesIO
from PIL import Image


class ImageEditAgent:
    def run(self, prompt: str, negative_prompt: str, image: Image.Image) -> bytes:
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise RuntimeError("STABILITY_API_KEY is missing in .env")
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        url = "https://api.stability.ai/v2beta/stable-image/transform/creative"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_format": "png",
            "strength": "0.65",
        }
        response = requests.post(
            url,
            headers=headers,
            files={"image": ("source.png", image_buffer, "image/png")},
            data=data,
            timeout=180,
        )
        if not response.ok:
            try:
                details = response.json()
            except Exception:
                details = response.text[:1200]
            raise RuntimeError(f"Stability AI edit error ({response.status_code}): {details}")
        return response.content
