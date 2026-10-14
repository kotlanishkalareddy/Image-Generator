import base64
import os
from io import BytesIO
from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import ImageEvaluation


def _image_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class ImageEvaluatorAgent:
    """Compares the newest output with the request and optional original, then suggests a refinement."""

    def __init__(self, model_name: str):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        ).with_structured_output(ImageEvaluation)

    def run(self, user_prompt: str, current_prompt: str, route: str,
            original_image: Image.Image | None, latest_image: Image.Image) -> ImageEvaluation:
        system = """
You are a vision-capable image quality evaluator and prompt optimizer.
Evaluate the LATEST GENERATED IMAGE against the user's request and current prompt.
If an ORIGINAL IMAGE is provided, also compare it with the latest image. For image-edit tasks,
identify whether requested changes happened and which important original details should be preserved.
For text-to-image tasks, judge prompt adherence, composition, artifacts, and visual coherence.
Do not claim certainty about hidden image details. Produce an improved prompt for the NEXT iteration.
When editing, write the optimized prompt so the latest image is the source for the next edit and the
original image is a preservation reference. Keep the user's requested changes central.
Return score 1-10, issues, comparison_notes, optimized_prompt, optimized_negative_prompt.
"""
        content = [
            {"type": "text", "text": (
                f"User request: {user_prompt or '[No written request]'}\n"
                f"Current prompt: {current_prompt}\nRoute: {route}\n"
                "Image A is the original source image if present. Image B is the latest generated result."
            )}
        ]
        if original_image is not None:
            content.append({"type": "text", "text": "Image A — ORIGINAL source image:"})
            content.append({"type": "image_url", "image_url": _image_data_url(original_image)})
        content.append({"type": "text", "text": "Image B — LATEST generated image to evaluate and optimize:"})
        content.append({"type": "image_url", "image_url": _image_data_url(latest_image)})
        return self.llm.invoke([SystemMessage(content=system), HumanMessage(content=content)])
