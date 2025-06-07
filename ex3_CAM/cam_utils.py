# c:\Users\kopno\Desktop\TP-S6-R6\cam_utils.py
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torchcam import methods
from torchcam.methods._utils import locate_candidate_layer
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

MODEL_CONFIGS = {
    "ResNet50": {
        "model_loader": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        "fc_layer": "fc",
    },
    "ResNet18": {
        "model_loader": lambda: models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "fc_layer": "fc",
    },
    "VGG16": {
        "model_loader": lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.VGG16_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.6",
    },
    "MobileNetV3_Small": {
        "model_loader": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.3",
    },
    "MobileNetV3_Large": {
        "model_loader": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.3",
    },
    "RegNet_Y_400MF": {
        "model_loader": lambda: models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.RegNet_Y_400MF_Weights.IMAGENET1K_V1,
        "fc_layer": "fc",
    },
    "ConvNeXt_Tiny": {
        "model_loader": lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.2",
    },
    "ConvNeXt_Small": {
        "model_loader": lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.2",
    },
    "EfficientNet_B0": {
        "model_loader": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.1",
    },
    "EfficientNet_B4": {
        "model_loader": lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1),
        "target_layer_getter": lambda model: locate_candidate_layer(model, (3, 224, 224)),
        "weights": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        "fc_layer": "classifier.1",
    },
}

CAM_ALGORITHMS = {
    "CAM": methods.CAM,
    "GradCAM": methods.GradCAM,
    "GradCAM++": methods.GradCAMpp,
    "SmoothGradCAM++": methods.SmoothGradCAMpp,
    "ScoreCAM": methods.ScoreCAM,
    "SSCAM": methods.SSCAM,
    "ISCAM": methods.ISCAM,
    "XGradCAM": methods.XGradCAM,
    "LayerCAM": methods.LayerCAM,
}

def load_model(model_name: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not configured.")
    model = MODEL_CONFIGS[model_name]["model_loader"]()
    model.eval() # Set model to evaluation mode
    return model

def get_model_preprocessing_transform(model_name: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not configured.")
    weights = MODEL_CONFIGS[model_name].get("weights")
    if weights:
        return weights.transforms()
    # Fallback generic preprocessing if specific weights object doesn't provide one
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image_pil: Image.Image, model_name: str) -> torch.Tensor:
    """Preprocesses a PIL image for a given model."""
    img_tensor = normalize(to_tensor(resize(image_pil, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img_tensor.unsqueeze(0) # Add batch dimension

def get_target_layer(model, model_name: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not configured.")
    return MODEL_CONFIGS[model_name]["target_layer_getter"](model)

def generate_cam_image(model, model_name: str, cam_method_name: str, input_tensor: torch.Tensor, original_image_pil: Image.Image, target_category_idx: int = None, custom_target_layer: str = None):
    if cam_method_name not in CAM_ALGORITHMS:
        raise ValueError(f"CAM method {cam_method_name} not supported.")
    
    # Use custom target layer if provided, otherwise use default
    if custom_target_layer:
        target_layer = custom_target_layer
    else:
        target_layer = get_target_layer(model, model_name)
    
    fc_layer = MODEL_CONFIGS[model_name]["fc_layer"]
    
    cam_algorithm_class = CAM_ALGORITHMS[cam_method_name]
    
    # Only pass fc_layer for CAM method that requires it
    if cam_method_name == "CAM":
        cam_extractor = cam_algorithm_class(model, target_layer=[target_layer], fc_layer=fc_layer)
    else:
        cam_extractor = cam_algorithm_class(model, target_layer=[target_layer])
    
    # Get model output
    output = model(input_tensor)
    
    # Determine target class
    if target_category_idx is None:
        target_category_idx = output.squeeze(0).argmax().item()
    
    # Generate CAM
    activation_maps = cam_extractor(target_category_idx, output)
    activation_map = activation_maps[0] if len(activation_maps) == 1 else cam_extractor.fuse_cams(activation_maps)
    
    # Prepare raw activation map (numpy array)
    raw_map_numpy = activation_map.squeeze().cpu().numpy() # Use squeeze() for robustness
    
    # Overlay CAM on the original image
    resized_img = resize(original_image_pil, (224, 224)) # Ensure image is resized for overlay
    # Convert single-channel activation map to PIL Image for overlay_mask
    activation_pil = to_pil_image(activation_map, mode="F")
    cam_image_overlay = overlay_mask(resized_img, activation_pil, alpha=0.6)
    
    return raw_map_numpy, cam_image_overlay

def get_prediction_score(model, input_tensor: torch.Tensor, target_category_idx: int):
    """Gets the model's confidence score for the target class and the predicted class."""
    with torch.no_grad(): # Ensure no gradients are computed during inference
        output = model(input_tensor)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    score_for_target_class = probabilities[target_category_idx].item()
    predicted_class_idx = torch.argmax(probabilities).item()
    
    return score_for_target_class, predicted_class_idx