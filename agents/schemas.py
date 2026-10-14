from typing import Literal
from pydantic import BaseModel, Field


class IntentDecision(BaseModel):
    route: Literal["text_to_image", "image_edit"] = Field(
        description="text_to_image creates a new scene; image_edit modifies an uploaded source image."
    )
    intent_summary: str
    positive_prompt: str
    negative_prompt: str = "blurry, low quality, distorted, artifacts"
    reason: str


class PromptReview(BaseModel):
    positive_prompt: str
    negative_prompt: str
    review_notes: str


class ImageEvaluation(BaseModel):
    score: int = Field(ge=1, le=10)
    issues: str
    comparison_notes: str
    optimized_prompt: str
    optimized_negative_prompt: str
