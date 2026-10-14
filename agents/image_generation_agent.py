import os
import requests


class TextToImageAgent:
    def run(self, prompt: str, negative_prompt: str, aspect_ratio: str) -> bytes:
        api_key = os.getenv("STABILITY_API_KEY")
        if not api_key:
            raise RuntimeError("STABILITY_API_KEY is missing in .env")
        url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }
        response = requests.post(url, headers=headers, files={"none": ("", b"")}, data=data, timeout=180)
        if not response.ok:
            raise RuntimeError(f"Stability AI generation error ({response.status_code}): {_error_detail(response)}")
        return response.content


def _error_detail(response):
    try:
        return response.json()
    except Exception:
        return response.text[:1200]
