import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

from .router_agent import IntentRouterAgent
from .prompt_critic_agent import PromptCriticAgent
from .image_generation_agent import TextToImageAgent
from .image_edit_agent import ImageEditAgent
from .image_evaluator_agent import ImageEvaluatorAgent

load_dotenv()


def plan_request(user_prompt, original_image, style, aspect_ratio):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    router = IntentRouterAgent(model_name)
    decision = router.run(user_prompt, original_image is not None, style, aspect_ratio)
    route = decision.route
    trace = [f"IntentRouterAgent selected {route}: {decision.reason}"]
    critic = PromptCriticAgent(model_name)
    review = critic.run(user_prompt, route, decision.positive_prompt, decision.negative_prompt)
    trace.append("PromptCriticAgent prepared a reviewed draft for human approval.")
    return {
        "route": route,
        "intent_summary": decision.intent_summary,
        "reason": decision.reason,
        "positive_prompt": review.positive_prompt,
        "negative_prompt": review.negative_prompt,
        "review_notes": review.review_notes,
        "trace": trace,
    }


def generate_from_approved_prompt(route, positive_prompt, negative_prompt, aspect_ratio,
                                  original_image, latest_image_bytes):
    trace_item = ""
    # Iterative optimization always edits the most recent result, not the original.
    if latest_image_bytes is not None:
        source = Image.open(BytesIO(latest_image_bytes)).convert("RGB")
        image_bytes = ImageEditAgent().run(positive_prompt, negative_prompt, source)
        return image_bytes, "image_edit", "ImageEditAgent used the LATEST generated image as the next iteration source."
    if route == "image_edit" and original_image is not None:
        image_bytes = ImageEditAgent().run(positive_prompt, negative_prompt, original_image)
        trace_item = "ImageEditAgent used the ORIGINAL uploaded image as the source."
        return image_bytes, "image_edit", trace_item
    image_bytes = TextToImageAgent().run(positive_prompt, negative_prompt, aspect_ratio)
    trace_item = "TextToImageAgent generated a new image from the human-approved prompt."
    return image_bytes, "text_to_image", trace_item


def evaluate_and_optimize(user_prompt, approved_prompt, original_image, latest_image_bytes, route):
    if not latest_image_bytes:
        raise ValueError("Generate an image before evaluation.")
    latest = Image.open(BytesIO(latest_image_bytes)).convert("RGB")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    evaluator = ImageEvaluatorAgent(model_name)
    result = evaluator.run(user_prompt, approved_prompt, route, original_image, latest)
    return {
        "score": result.score,
        "issues": result.issues,
        "comparison_notes": result.comparison_notes,
        "optimized_prompt": result.optimized_prompt,
        "optimized_negative_prompt": result.optimized_negative_prompt,
    }
