import imgaug.augmenters as iaa
import numpy as np


def apply_brightness(image_np: np.ndarray) -> list[np.ndarray]:
    """Applies brightness transformation.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        if index < 5:
            value = 0.01 * (3**index)
        else:
            value = 1.5 * (2**(index - 5))
        aug = iaa.Multiply(value)
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
                
    return results


def apply_contrast(image_np: np.ndarray) -> list[np.ndarray]:
    """Applies contrast transformation.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        if index < 5:
            value = 0.01 * (3**index)
        else:
            value = 1.5 * (2**(index - 5))
        
        aug = iaa.LinearContrast(value)
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
    
    return results


def apply_shear(image_np: np.ndarray, axis: str) -> list[np.ndarray]:
    """Applies shear transformation.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        degrees = 0.1 + (index**2.05)
        if axis == 'x':
            aug = iaa.ShearX((degrees, degrees))
        else:
            aug = iaa.ShearY((degrees, degrees))
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
    
    return results


def apply_scale(image_np: np.ndarray) -> list[np.ndarray]:
    """Applies scaling transformation.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        scale = 0.1 * (1.75 ** index)
        aug = iaa.Affine(scale=scale)
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
    
    return results


def apply_elastic_distortion(image_np: np.ndarray) -> list[np.ndarray]:
    """Applies elastic distortion.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        alpha = (1 + index) * 10
        sigma = 10/(index+1)
        aug = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
    
    return results


def apply_perspective_transform(image_np: np.ndarray) -> list[np.ndarray]:
    """Applies perspective transformation.
    """
    results = []
    original_image_np = image_np.copy()
    for index in range(10):
        scale = np.random.uniform(0.1, 0.3)
        aug = iaa.PerspectiveTransform(scale=scale)
        augmented_image_np = aug.augment_image(original_image_np)
        results.append(augmented_image_np)
    
    return results


AUGMENTER_FUNCTIONS = {
    "Brightness": apply_brightness,
    "Contrast": apply_contrast,
    "ShearX": apply_shear,
    "ShearY": apply_shear,
    "Scale": apply_scale,
    "Elastic Distortion": apply_elastic_distortion,
    "Perspective Transform": apply_perspective_transform
}

SELECTED_TRANSFORMATIONS_FOR_APP = ["Brightness", "ShearX", "Scale", "Elastic Distortion", "Perspective Transform", "Contrast", "ShearY"]