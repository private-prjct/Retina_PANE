import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from retina_breaker import RetinaCloakEngine
import io

st.set_page_config(page_title="RetinaNet Cloaking Lab", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.45

@st.cache_resource
def load_engine():
    return RetinaCloakEngine(device=device)

engine = load_engine()

def pil_to_tensor(pil_img):
    return F.to_tensor(pil_img).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    return F.to_pil_image(tensor.squeeze(0).cpu())

def image_bytes(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

st.title("RetinaNet Adversarial Cloaking")
st.caption("Object detection suppression using RetinaNet ResNet50 FPN")

uploaded_file = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

steps = st.sidebar.slider("Optimization steps", 5, 100, 40)
epsilon = st.sidebar.slider("Epsilon", 0.001, 0.1, 0.03, format="%.3f")
alpha = st.sidebar.slider("Alpha", 0.001, 0.01, 0.002, format="%.4f")

st.sidebar.text(f"Device: {device.upper()}")

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    input_tensor = pil_to_tensor(img_pil)

    with st.spinner("Running baseline detection"):
        base_results = engine.run_inference(input_tensor)
        base_scores = base_results["scores"]
        base_count = (base_scores > CONF_THRESH).sum().item()

    st.info(f"Baseline detections above {CONF_THRESH}: {base_count}")

    if st.sidebar.button("Generate cloaked image"):
        with st.spinner("Running adversarial optimization"):
            cloaked_tensor = engine.apply_cloak(
                input_tensor,
                steps=steps,
                alpha=alpha,
                epsilon=epsilon
            )

        with st.spinner("Running detection on cloaked image"):
            cloak_results = engine.run_inference(cloaked_tensor)
            cloak_scores = cloak_results["scores"]
            cloak_count = (cloak_scores > CONF_THRESH).sum().item()

        cloaked_pil = tensor_to_pil(cloaked_tensor)

        st.subheader("Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_pil, use_container_width=True)
            st.metric("Detections", base_count)

        with col2:
            st.image(cloaked_pil, use_container_width=True)
            delta = cloak_count - base_count
            st.metric("Detections", cloak_count, delta=delta)

        if base_count > 0:
            reduction = base_count - cloak_count
            pct = reduction / base_count * 100
            if reduction > 0:
                st.success(f"Detection reduction: {reduction} ({pct:.1f}%)")
            elif reduction == 0:
                st.warning("No reduction observed")
            else:
                st.error("Detection count increased")

        st.subheader("Download cloaked image")
        st.download_button(
            "Download PNG",
            data=image_bytes(cloaked_pil),
            file_name="cloaked.png",
            mime="image/png"
        )
    else:
        st.image(img_pil, use_container_width=True)
else:
    st.info("Upload an image to begin")
