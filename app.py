import streamlit as st
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

from agents.orchestrator import plan_request, generate_from_approved_prompt, evaluate_and_optimize

load_dotenv()
st.set_page_config(page_title="LangChain Image Studio", page_icon="🧠", layout="wide")
st.title("🧠 Multi-Agent Image Studio")
st.caption("Intent routing, human approval, image comparison, and iterative optimization.")

# Persist workflow artifacts across Streamlit reruns.
defaults = {
    "plan": None,
    "approved_prompt": "",
    "approved_negative": "",
    "route": None,
    "original_image_bytes": None,
    "latest_image_bytes": None,
    "iteration": 0,
    "evaluation": None,
    "trace": [],
    "last_aspect_ratio": "1:1",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.header("Generation settings")
    style = st.selectbox("Style", ["Auto", "Photorealistic", "Digital art", "Anime", "Watercolor", "3D render"])
    aspect_ratio = st.selectbox("Aspect ratio for new images", ["1:1", "16:9", "9:16", "4:3", "3:2"])
    max_iterations = st.slider("Optimization rounds", 1, 3, 2)
    st.caption("API keys are read from .env.")

prompt = st.text_area(
    "Describe what you want",
    placeholder="Create a cinematic image of a futuristic city at sunset. Or upload an image and ask to change its background.",
    height=110,
    key="user_prompt",
)
upload = st.file_uploader("Optional original/reference image", type=["png", "jpg", "jpeg", "webp"])

if upload is not None:
    try:
        image = Image.open(upload).convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        # Preserve the uploaded original once per newly selected upload.
        if st.session_state.get("uploaded_file_name") != upload.name:
            st.session_state.original_image_bytes = buf.getvalue()
            st.session_state.latest_image_bytes = None
            st.session_state.plan = None
            st.session_state.evaluation = None
            st.session_state.iteration = 0
            st.session_state.trace = []
            st.session_state.uploaded_file_name = upload.name
        st.image(image, caption="Original image", width=320)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")

col1, col2 = st.columns(2)
with col1:
    if st.button("1 · Plan request", type="primary", use_container_width=True):
        if not prompt.strip() and st.session_state.original_image_bytes is None:
            st.error("Enter a prompt or upload an image.")
        else:
            try:
                original = (
                    Image.open(BytesIO(st.session_state.original_image_bytes)).convert("RGB")
                    if st.session_state.original_image_bytes else None
                )
                with st.spinner("Gemini is routing the request and drafting a prompt..."):
                    plan = plan_request(prompt.strip(), original, style, aspect_ratio)
                st.session_state.plan = plan
                st.session_state.route = plan["route"]
                st.session_state.approved_prompt = plan["positive_prompt"]
                st.session_state.approved_negative = plan["negative_prompt"]
                st.session_state.evaluation = None
                st.session_state.trace = plan["trace"]
                st.success(f"Plan ready. Suggested route: {plan['route']}")
            except Exception as exc:
                st.error(f"Planning failed: {exc}")

with col2:
    if st.button("Reset workflow", use_container_width=True):
        for key in list(defaults.keys()) + ["uploaded_file_name"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.plan:
    st.divider()
    st.subheader("Human-in-the-loop approval")
    st.write("Review and edit the prompts below. Image generation will not start until you approve them.")
    st.markdown(f"**Proposed route:** `{st.session_state.route}`")
    st.text_area("Final positive prompt (editable)", key="approved_prompt", height=150)
    st.text_area("Negative prompt (editable)", key="approved_negative", height=90)
    st.write("**Gemini's interpretation:**", st.session_state.plan.get("intent_summary", ""))
    st.write("**Why this route was selected:**", st.session_state.plan.get("reason", ""))

    if st.button("2 · Approve prompt and generate", type="primary", use_container_width=True):
        if not st.session_state.approved_prompt.strip():
            st.error("The positive prompt cannot be empty.")
        else:
            try:
                with st.spinner("Calling the selected image agent..."):
                    original = (
                        Image.open(BytesIO(st.session_state.original_image_bytes)).convert("RGB")
                        if st.session_state.original_image_bytes else None
                    )
                    image_bytes, route_used, trace_item = generate_from_approved_prompt(
                        route=st.session_state.route,
                        positive_prompt=st.session_state.approved_prompt,
                        negative_prompt=st.session_state.approved_negative,
                        aspect_ratio=aspect_ratio,
                        original_image=original,
                        latest_image_bytes=None,
                    )
                st.session_state.latest_image_bytes = image_bytes
                st.session_state.route = route_used
                st.session_state.iteration = 0
                st.session_state.evaluation = None
                st.session_state.last_aspect_ratio = aspect_ratio
                st.session_state.trace.append(trace_item)
                st.success("Image generated. Review it below.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

if st.session_state.latest_image_bytes:
    st.divider()
    st.subheader("Current result")
    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.original_image_bytes:
            st.image(st.session_state.original_image_bytes, caption="Original image", use_container_width=True)
        else:
            st.info("No original image was uploaded.")
    with c2:
        st.image(st.session_state.latest_image_bytes, caption=f"Latest generated image · round {st.session_state.iteration}", use_container_width=True)
        st.download_button(
            "Download latest image",
            data=st.session_state.latest_image_bytes,
            file_name=f"image_round_{st.session_state.iteration}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.subheader("Evaluate and optimize")
    st.write(
        "The evaluator compares the latest result with your request and, when available, "
        "the original image. It suggests a revised prompt; you remain in control of whether to use it."
    )
    can_optimize = st.session_state.iteration < max_iterations
    if st.button("3 · Evaluate latest image and suggest improvements", disabled=not can_optimize):
        try:
            original = (
                Image.open(BytesIO(st.session_state.original_image_bytes)).convert("RGB")
                if st.session_state.original_image_bytes else None
            )
            with st.spinner("Gemini is evaluating the original (if any) and latest result..."):
                evaluation = evaluate_and_optimize(
                    user_prompt=st.session_state.get("user_prompt", ""),
                    approved_prompt=st.session_state.approved_prompt,
                    original_image=original,
                    latest_image_bytes=st.session_state.latest_image_bytes,
                    route=st.session_state.route,
                )
            st.session_state.evaluation = evaluation
            st.session_state.trace.append(
                f"ImageEvaluatorAgent evaluated round {st.session_state.iteration}: {evaluation['score']}/10."
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    if st.session_state.evaluation:
        ev = st.session_state.evaluation
        st.markdown(f"**Quality score:** {ev['score']}/10")
        st.markdown("**What could improve:**")
        st.write(ev["issues"])
        st.markdown("**Evaluator's reasoning:**")
        st.write(ev["comparison_notes"])
        st.text_area("Suggested optimized prompt (editable)", value=ev["optimized_prompt"], key="suggested_prompt", height=140)
        st.text_area("Suggested negative prompt (editable)", value=ev["optimized_negative_prompt"], key="suggested_negative", height=80)
        st.caption("Human approval is required before another image is generated.")
        if st.button("4 · Approve optimization and generate next version", type="primary", use_container_width=True, disabled=not can_optimize):
            if not st.session_state.suggested_prompt.strip():
                st.error("The optimized prompt cannot be empty.")
            else:
                try:
                    with st.spinner("Generating an optimized version from the latest image..."):
                        original = (
                            Image.open(BytesIO(st.session_state.original_image_bytes)).convert("RGB")
                            if st.session_state.original_image_bytes else None
                        )
                        image_bytes, route_used, trace_item = generate_from_approved_prompt(
                            route="image_edit",
                            positive_prompt=st.session_state.suggested_prompt,
                            negative_prompt=st.session_state.suggested_negative,
                            aspect_ratio=st.session_state.last_aspect_ratio,
                            original_image=original,
                            latest_image_bytes=st.session_state.latest_image_bytes,
                        )
                    # Keep original_image_bytes unchanged; the latest image becomes the next iteration's source.
                    st.session_state.latest_image_bytes = image_bytes
                    st.session_state.approved_prompt = st.session_state.suggested_prompt
                    st.session_state.approved_negative = st.session_state.suggested_negative
                    st.session_state.iteration += 1
                    st.session_state.evaluation = None
                    st.session_state.trace.append(trace_item)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Optimization generation failed: {exc}")

with st.expander("Agent trace", expanded=False):
    if st.session_state.trace:
        for event in st.session_state.trace:
            st.write(f"• {event}")
    else:
        st.caption("The trace will appear after you plan a request.")
