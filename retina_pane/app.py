import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F
from retina_breaker import RetinaCloakEngine
import io

# --- Setup ---
st.set_page_config(page_title="RetinaNet Cloaking Lab", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_engine():
    return RetinaCloakEngine(device=device)

engine = load_engine()

def pil_to_tensor(pil_img):
    return F.to_tensor(pil_img).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    return F.to_pil_image(tensor.squeeze(0).cpu())

def get_image_download_link(img_pil, filename="cloaked_image.png"):
    """Generate download button for PIL image"""
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# --- UI Layout ---
st.title(" RetinaNet Adversarial Cloaking")
st.caption("A research-oriented tool to evaluate object detection robustness using RetinaNet-ResNet50-FPN-V2.")

# Sidebar controls
uploaded_file = st.sidebar.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.subheader("Attack Parameters")
steps = st.sidebar.slider("Optimization Steps", 5, 100, 40, help="More steps = stronger attack but slower")
epsilon = st.sidebar.slider("Perturbation Limit (ε)", 0.001, 0.1, 0.03, format="%.3f", 
                            help="Maximum pixel change (L∞ norm)")
alpha = st.sidebar.slider("Step Size (α)", 0.001, 0.01, 0.002, format="%.4f",
                          help="Learning rate per iteration")

st.sidebar.markdown("---")
st.sidebar.info(f"**Device:** {device.upper()}")

if uploaded_file:
    # 1. Prepare Images
    img_pil = Image.open(uploaded_file).convert("RGB")
    input_tensor = pil_to_tensor(img_pil)

    # 2. Run Baseline Detection
    with st.spinner("Analyzing original image..."):
        base_results = engine.run_inference(input_tensor)
        base_count = len(base_results['scores'])

    # Display baseline info
    st.info(f"**Baseline Detection:** {base_count} objects detected with confidence > 0.3")

    # 3. Apply Cloak
    if st.sidebar.button(" Generate Cloaked Image", type="primary"):
        with st.spinner(f"Applying adversarial perturbations ({steps} iterations)..."):
            progress_bar = st.progress(0)
            
            # Apply cloaking with progress callback
            cloaked_tensor = engine.apply_cloak(
                input_tensor, 
                steps=steps, 
                alpha=alpha, 
                epsilon=epsilon
            )
            
            # Run detection on cloaked image
            cloak_results = engine.run_inference(cloaked_tensor)
            cloak_count = len(cloak_results['scores'])
            cloaked_pil = tensor_to_pil(cloaked_tensor)
            
            progress_bar.progress(100)

        # 4. Display Results Side-by-Side
        st.markdown("---")
        st.subheader(" Results Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(img_pil, use_container_width=True)
            st.metric("Objects Detected", base_count)
            
        with col2:
            st.markdown("**Cloaked Image**")
            st.image(cloaked_pil, use_container_width=True)
            reduction = base_count - cloak_count
            st.metric(
                "Objects Detected", 
                cloak_count, 
                delta=f"-{reduction}" if reduction > 0 else f"+{abs(reduction)}",
                delta_color="inverse"
            )
        
        # Attack effectiveness summary
        if reduction > 0:
            effectiveness = (reduction / base_count * 100) if base_count > 0 else 0
            st.success(f" Attack reduced detections by {reduction} ({effectiveness:.1f}% reduction)")
        elif reduction == 0:
            st.warning(" Attack had no effect. Try increasing steps or epsilon.")
        else:
            st.error(" Attack increased detections (unexpected behavior)")
        
        # Download button
        st.markdown("---")
        st.subheader(" Download Cloaked Image")
        
        img_bytes = get_image_download_link(cloaked_pil)
        st.download_button(
            label="Download Cloaked Image (PNG)",
            data=img_bytes,
            file_name="cloaked_adversarial.png",
            mime="image/png"
        )
            
    else:
        # Show original image before attack
        st.image(img_pil, caption="Original Image - Click 'Generate Cloaked Image' to start attack", 
                use_container_width=True)
        
else:
    st.info(" Please upload an image in the sidebar to begin the cloaking process.")
    