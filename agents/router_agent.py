import os
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import IntentDecision


class IntentRouterAgent:
    def __init__(self, model_name: str):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        ).with_structured_output(IntentDecision)

    def run(self, user_prompt: str, has_image: bool, style: str, aspect_ratio: str) -> IntentDecision:
        system = """
You are an intent-routing and prompt-planning agent.
Choose text_to_image when the user asks for a new image/scene. Choose image_edit when an uploaded image
should be modified. If there is no uploaded image, never choose image_edit.
Create a visual prompt that follows the request. If an uploaded image exists, use it as context.
When editing, preserve important subject identity, layout, and details unless the user asks to change them.
Return only the required structured fields.
"""
        request = f"""
User request: {user_prompt or '[No text prompt provided]'}
Image uploaded: {has_image}
Requested style: {style}
Aspect ratio for new image: {aspect_ratio}
"""
        return self.llm.invoke([("system", system), ("human", request)])
