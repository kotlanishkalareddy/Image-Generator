# LangChain Multi-Agent Image Studio with Human-in-the-Loop

## Features
- Streamlit UI accepts a text prompt, an optional original image, or both.
- Gemini/LangChain intent router chooses `text_to_image` or `image_edit`.
- Prompt critic creates a draft; the human reviews and edits positive/negative prompts before generation.
- Dedicated Stability AI agents generate a new image or transform a source image.
- Vision-capable Gemini evaluator compares the latest output to the user's request and, if available, the original source image.
- The evaluator proposes an optimized prompt, but the human must approve/edit it before the next round.
- Each optimization uses the **latest generated image** as the next edit source while retaining the original image unchanged for comparison.
- Optimization rounds are bounded in the sidebar to avoid endless generation.

## Workflow
1. Enter a request and optionally upload an original image.
2. Click **Plan request**.
3. Review/edit prompts, then click **Approve prompt and generate**.
4. Inspect the original and latest output side by side when an original exists.
5. Click **Evaluate latest image and suggest improvements**.
6. Review/edit the optimized prompt and explicitly approve the next iteration.
7. Repeat until satisfied or the configured iteration limit is reached.

## Setup
Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add `GEMINI_API_KEY`, `STABILITY_API_KEY`, and optionally `GEMINI_MODEL` to `.env`, then:

```bash
streamlit run app.py
```

Windows PowerShell: use `.venv\\Scripts\\Activate.ps1` and `copy .env.example .env`.

## API assumption
The implementation uses Stability AI hosted endpoints:
- `/v2beta/stable-image/generate/core`
- `/v2beta/stable-image/transform/creative`

Check the current endpoint documentation and account access if the provider returns a validation error. Creative transform can alter more than the exact requested area; this MVP does not perform masked inpainting.

## Notes
- The original upload is retained separately from generated versions.
- On each approved optimization, the latest image becomes the source for the next transform.
- The original is shown and sent to the vision evaluator as a preservation/comparison reference where available.
- Keep keys in `.env`; never commit secrets.
- This is an MVP; add authentication, moderation, retries, observability, and automated tests before production.
