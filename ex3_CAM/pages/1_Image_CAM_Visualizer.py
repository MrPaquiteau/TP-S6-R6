# c:\Users\kopno\Desktop\TP-S6-R6\pages\1_Image_CAM_Visualizer.py
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt # Import matplotlib
import requests

# Adjust import path if necessary based on your project structure
# Assuming cam_utils is in the parent directory of 'pages'
import sys
sys.path.append("..") # Add parent directory to sys.path
from cam_utils import (
    load_model, preprocess_image, generate_cam_image, 
    get_prediction_score, get_target_layer, MODEL_CONFIGS, CAM_ALGORITHMS
)

LABEL_MAP = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
).json()

class_choices = [f"{idx} - {class_name}" for idx, class_name in enumerate(LABEL_MAP)]

def visualization_page():    
    st.header("👁️ Image CAM Visualizer")
    st.markdown("Upload an image, select a model, a CAM method, and a target class to visualize the Class Activation Map.")

    col_config, col_results = st.columns([1, 2]) # Configuration column, Results display column

    with col_config:
        st.subheader("Configuration")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="vis_uploader")
        
        model_names = list(MODEL_CONFIGS.keys())
        selected_model_name = st.selectbox("Select Model:", model_names, key="vis_model")

        cam_method_names = list(CAM_ALGORITHMS.keys())
        selected_cam_method = st.selectbox("Select CAM Method:", cam_method_names, key="vis_cam")
        
        custom_layer = "" # Initialize custom_layer
        if selected_model_name:
            # Load model temporarily to get default layer
            temp_model = load_model(selected_model_name)
            default_layer = get_target_layer(temp_model, selected_model_name)
            custom_layer = st.text_input(
                "Target Layer (optional):", 
                value="", 
                placeholder=f"Default: {default_layer}",
                key="vis_custom_layer",
                help="Leave empty to use default layer, or specify custom layer name"
            )
        
        # Sélection de la classe cible comme dans app.py
        class_selection = st.selectbox(
            "Class selection",
            ["Predicted class (argmax)"] + class_choices,
            key="vis_class_selection"
        )
        if class_selection == "Predicted class (argmax)":
            target_class_idx = None
            label_str = ""
        else:
            target_class_idx = int(class_selection.split(" - ")[0])
            label_str = LABEL_MAP[target_class_idx]
    
        if uploaded_file is None:
            st.info("Please upload an image to start.")
        else:
            # Display a small preview of the uploaded image in the config column
            # This image instance is just for preview in config
            preview_image_pil = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image_pil, caption="Uploaded Image Preview", use_container_width=True)

    if uploaded_file is not None and selected_model_name and selected_cam_method:
        # Place button in config column or make it span if preferred
        with col_config:
            generate_button_pressed = st.button(
                f"Generate {selected_cam_method} for {selected_model_name}", 
                key="vis_generate_cam",
                use_container_width=True
            )

        if generate_button_pressed:
            try:
                # Load the image for processing. This ensures we use the most current uploaded file.
                current_original_image_pil = Image.open(uploaded_file).convert("RGB")
                
                with st.spinner("Loading model and generating CAM..."):
                    model = load_model(selected_model_name)
                    input_tensor = preprocess_image(current_original_image_pil, selected_model_name)

                    # Si target_class_idx est None, get_prediction_score utilisera argmax
                    score, predicted_class_idx = get_prediction_score(
                        model, input_tensor, 
                        target_class_idx if target_class_idx is not None else None
                    )
                    
                    target_layer_to_use = custom_layer.strip() if custom_layer.strip() else None
                    
                    # generate_cam_image now returns raw_map_numpy and cam_overlay_pil
                    raw_map_numpy, cam_overlay_pil = generate_cam_image(
                        model, selected_model_name, selected_cam_method, 
                        input_tensor, current_original_image_pil, 
                        target_category_idx=target_class_idx,
                        custom_target_layer=target_layer_to_use
                    )
                with col_results:
                    st.subheader("CAM Visualizations")
                    
                    viz_col_orig, viz_col_raw, viz_col_overlay = st.columns(3)

                    with viz_col_orig:
                        st.image(current_original_image_pil, caption="Original Image", use_container_width=True)
                    
                    with viz_col_raw:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(raw_map_numpy)
                        ax.axis("off")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        st.caption("Raw Activation Map")

                    with viz_col_overlay:
                        layer_info = f" (Layer: {target_layer_to_use})" if target_layer_to_use else ""
                        label_str = LABEL_MAP[target_class_idx] if target_class_idx is not None and 0 <= target_class_idx < len(LABEL_MAP) else ""
                        st.image(
                            cam_overlay_pil, 
                            caption=f"{selected_cam_method} on {selected_model_name}{layer_info} (Target: {target_class_idx if target_class_idx is not None else 'Predicted'} - {label_str})", 
                            use_container_width=True
                        )
                    
                    st.write("---")
                    st.write(f"**Model Performance:**")
                    st.write(f"- Predicted Class Index: `{predicted_class_idx}`")
                    if 0 <= predicted_class_idx < len(LABEL_MAP):
                        st.write(f"- Predicted Label: `{LABEL_MAP[predicted_class_idx]}`")
                    st.write(f"- Score for Target Class ({target_class_idx if target_class_idx is not None else predicted_class_idx}): `{score:.4f}`")
            
            except Exception as e:
                with col_results:
                    st.error(f"An error occurred: {e}")
                    st.error("Please check model compatibility, CAM method, target class index, and layer name.")
    
if __name__ == "__main__":
    visualization_page()