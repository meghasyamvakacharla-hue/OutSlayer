import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Offroad Image Segmentation", layout="centered", page_icon="🚜")

# Custom UI aesthetics for the modern web vibe requested
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .stButton>button {
        background-color: #2e6bba;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #4a8cdb;
        transform: scale(1.02);
    }
    .sidebar .sidebar-content {
        background: #1E232F;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚜 Offroad Autonomous Mapping AI")
st.markdown("**Welcome to the interactive segmentation website!** Upload an image below to see our MobileNetV3 AI classify dirt, roads, vehicles, and sky in real-time.")

@st.cache_resource
def load_model():
    """Loads the best model state into memory once."""
    num_classes = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load("best_model_20_epochs.pth", map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, None

model, device = load_model()

if model is None:
    st.error("Model weights NOT found! Please finish your training run which produces `best_model_20_epochs.pth`.")
else:
    st.success(f"Model successfully loaded. Running inference on: {device.type.upper()}")

# Define colors for overlay based heavily on exact RGB distribution of the dataset masks
color_map = {
    0: [0, 0, 0],         # Background -> Black
    1: [34, 139, 34],     # Vegetation/Shadows -> Green
    2: [205, 133, 63],    # Dirt/Terrain -> Brown/Orange
    3: [244, 164, 96],    # Sandy Dirt -> SandyBrown
    4: [128, 128, 128],   # Road/Path -> Gray
    5: [135, 206, 235]    # Sky -> Light Blue
}

uploaded_file = st.file_uploader("Upload an Offroad Terrain Image (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None and model is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_img, use_container_width=True)
        
    with st.spinner("Neural Engine Segmenting..."):
        # Preprocess
        input_tensor = TF.to_tensor(TF.resize(original_img, (256, 256))).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)['out']
            predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy() # [256, 256]
            
        # Create Color Overlay
        overlay = np.zeros((256, 256, 3), dtype=np.uint8)
        for class_idx, color in color_map.items():
            overlay[predictions == class_idx] = color
            
        overlay_pil = Image.fromarray(overlay)
        # Resize overlay back to original dimensions for comparison
        overlay_upscaled = overlay_pil.resize(original_img.size, Image.NEAREST)
        
        # Soft Blender
        blended = Image.blend(original_img, overlay_upscaled, alpha=0.55)
        
    with col2:
        st.subheader("AI Prediction Overlay")
        st.image(blended, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Legend")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown("⬛ Background")
    c2.markdown("🟩 Vegetation")
    c3.markdown("🟫 Dirt/Terrain")
    c4.markdown("🟨 Sandy Dirt")
    c5.markdown("⬜ Road/Path")
    c6.markdown("🟦 Sky")
