# c:\Users\kopno\Desktop\TP-S6-R6\pages\2_Augmentation_Analyzer.py
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os # Added for path operations

import sys
sys.path.append("..")
from cam_utils import (
    load_model, preprocess_image, generate_cam_image, 
    get_prediction_score, MODEL_CONFIGS, CAM_ALGORITHMS
)
from image_augmenters import AUGMENTER_FUNCTIONS, SELECTED_TRANSFORMATIONS_FOR_APP

EXPORT_DIR = "CAM/ex3/exports" # Directory to save augmented images and CAMs

def augmentation_analyzer_page():
    st.header("🔬 Augmentation Impact Analyzer")
    st.markdown("""
    Apply a chosen transformation to an image, generating 10 variants with increasing intensity. 
    For each variant, visualize its CAM and see the model's confidence score for a target class.
    Results are compiled into a CSV file.
    """)

    st.sidebar.header("⚙️ Analysis Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"], key="aug_uploader")

    if not uploaded_file:
        st.info("👈 Please upload an image using the sidebar to begin.")
        return
    
    original_image_pil = Image.open(uploaded_file).convert("RGB")
    original_image_np = np.array(original_image_pil) # Convert PIL to NumPy for augmenters

    # Use the pre-selected list of 2 transformations for the dropdown
    selected_transform_name = st.sidebar.selectbox(
        "Select Transformation Type:",
        options=SELECTED_TRANSFORMATIONS_FOR_APP, 
        key="aug_transform_type",
        help=f"Choose one of the {len(SELECTED_TRANSFORMATIONS_FOR_APP)} available transformations."
    )

    model_names = list(MODEL_CONFIGS.keys())
    selected_model_name = st.sidebar.selectbox("Select Model:", model_names, key="aug_model")

    cam_method_names = list(CAM_ALGORITHMS.keys())
    selected_cam_method = st.sidebar.selectbox("Select CAM Method:", cam_method_names, key="aug_cam")

    target_class_idx = st.sidebar.number_input(
        "Target Class Index (ImageNet):", 
        value=242, min_value=0, max_value=999, 
        key="aug_target_class",
        help="e.g., 242 for 'golden retriever'"
    )

    st.subheader("Original Image")
    st.image(original_image_pil, caption="Original Image", width=300)

    if st.sidebar.button("🚀 Generate & Analyze Augmented Images", key="aug_generate_button"):
        if selected_transform_name not in AUGMENTER_FUNCTIONS:
            st.error(f"Transformation '{selected_transform_name}' is not implemented correctly.")
            return

        augment_function = AUGMENTER_FUNCTIONS[selected_transform_name]
        print(f"Using augmentation function: {augment_function.__name__}")
        
        try:
            model = load_model(selected_model_name)
        except Exception as e:
            st.error(f"Error loading model {selected_model_name}: {e}")
            return

        # Create export directory
        os.makedirs(EXPORT_DIR, exist_ok=True)
        st.info(f"Augmented images and CAMs will be saved to the '{EXPORT_DIR}/' directory.")

        results_data = []
        
        st.subheader(f"Augmented Images ({selected_transform_name}) & CAMs ({selected_cam_method} for class {target_class_idx})")
        
        num_cols = 5 # Display images in 2 rows of 5
        cols = st.columns(num_cols)
        
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        if selected_transform_name == "ShearX":
            images = augment_function(original_image_np, axis='x')
        elif selected_transform_name == "ShearY":
            images = augment_function(original_image_np, axis='y')
        else:
            images = augment_function(original_image_np)
            
        for idx, image_np in enumerate(images):
            # Preprocess image for model input
            image_pil = Image.fromarray(image_np)
            preprocessed_img = preprocess_image(image_pil, selected_model_name)

            # Get prediction score for the target class
            score = get_prediction_score(model, preprocessed_img, target_class_idx)

            # Display augmented image and score
            with cols[idx % num_cols]:
                st.image(image_pil, caption=f"Transformation : {idx+1}", width=140)
                st.markdown(f"**Score:** {score[0]:.4f}")

            # Save augmented image and CAM
            aug_img_path = os.path.join(EXPORT_DIR, f"{selected_transform_name.lower().replace(' ','_')}_{idx+1}.png")
            Image.fromarray(image_np).save(aug_img_path)

            # Collect results for CSV
            results_data.append({
                "Augmentation Index": idx+1,
                "Transformation": selected_transform_name,
                "Model": selected_model_name,
                "CAM Method": selected_cam_method,
                "Target Class": target_class_idx,
                "Score": f'{score[0]:.4f}',
                'Predicted Class':score[-1],
                "Augmented Image Path": aug_img_path
            })

            progress_bar.progress((idx+1)/len(images))
            status_text.text(f"Processed {idx+1}/{len(images)} images")

            # After loop, export results to CSV
            if len(results_data) == len(images):
                df = pd.DataFrame(results_data)
                csv_path = os.path.join(EXPORT_DIR, f"augmentation_analysis_results_{selected_transform_name.lower().replace(' ','_')}.csv")
                df.to_csv(csv_path, index=False)
                st.success(f"Results exported to {csv_path}")


if __name__ == "__main__":
    augmentation_analyzer_page()