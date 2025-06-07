# c:\Users\kopno\Desktop\TP-S6-R6\🏠_Home.py
import streamlit as st

st.set_page_config(
    page_title="CAM & Augmentation Explorer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🖼️ CAM & Image Augmentation Explorer")

st.markdown("""
Welcome! This application helps you explore Class Activation Maps (CAMs) and analyze the impact of image augmentations.

**Use the sidebar to navigate between tools:**
- **Image CAM Visualizer**: Upload an image, select a model and CAM method to see the visualization.
- **Augmentation Analyzer**: Apply transformations to an image, generate 10 variants, and analyze their model scores.
""")

st.sidebar.success("Select a page above.")

st.info("""
**Quick Start:**
1. Go to **Image CAM Visualizer** to test CAMs on a single image.
2. Go to **Augmentation Analyzer** to study how transformations affect model perception and CAMs.
   - You'll be able to choose one of the pre-selected transformations (e.g., Brightness, ShearX).
   - 10 augmented versions of your image will be generated.
   - CAMs and model confidence scores for a target class will be displayed for each.
   - Results can be downloaded as a CSV file.

**Dependencies**: `streamlit`, `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `imgaug`.
Make sure these are installed in your Python environment.
""")