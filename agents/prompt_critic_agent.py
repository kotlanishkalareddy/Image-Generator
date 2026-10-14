import os
from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import PromptReview


class PromptCriticAgent:
    def __init__(self, model_name: str):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.15,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        ).with_structured_output(PromptReview)

    def run(self, user_prompt: str, route: str, positive_prompt: str, negative_prompt: str) -> PromptReview:
        system = """
You are a Stable Diffusion prompt critic. Improve visual clarity and consistency while preserving the
user's intent. Do not add unrelated objects or contradict the request. Keep prompts concise and concrete.
"""
        message = f"""
Route: {route}
Original request: {user_prompt or '[No text prompt]'}
Draft positive prompt: {positive_prompt}
Draft negative prompt: {negative_prompt}
Return the final positive prompt, negative prompt, and brief review notes.
"""
        return self.llm.invoke([("system", system), ("human", message)])
