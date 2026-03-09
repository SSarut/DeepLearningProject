import torch
import os
import json
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from app.models.resnet import ResNet34_Gradual_Unfreezing
from app.models.mobilenetv3small import MobileNetV3Small
import cv2
import numpy as np
from app.augment.card_augmentation import (
    augment_rotate_scale_noise,
    augment_3d_warp_noise,
    augment_rotate_partial,
    augment_3d_partial,
    augment_3d_partialv2,
    augment_3d_warp_noisev2,
    augment_colour_stressv2,
    augment_rotate_partialv2,
    augment_rotate_scale_noisev2
)

MODELS_CONFIG = {
    "ResNet34DigitalCardOnly": {
        "class": ResNet34_Gradual_Unfreezing,
        "weight_path": os.path.join(os.path.dirname(__file__), "weights", "resnet34_v1_best.pth"),
        "num_classes": 9
    },
    "MobileNetV3SmallIRLDomainAdapt": {
        "class": MobileNetV3Small,
        "weight_path": os.path.join(os.path.dirname(__file__), "weights", "digital_to_irl_95.pth"),
        "num_classes": 9
    }
}

def get_model(model_name: str):
    if model_name not in MODELS_CONFIG:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    config = MODELS_CONFIG[model_name]
    model_class = config["class"]
    weight_path = config["weight_path"]
    num_classes = config["num_classes"]
    
    model = model_class(num_classes=num_classes)
    
    weight_dict = torch.load(weight_path, map_location='cpu')
    if 'model_state_dict' in weight_dict:
        weight_dict = weight_dict['model_state_dict']
    
    model.load_state_dict(weight_dict, strict=True)
    model.eval()
    return model

def image_preprocess(image_path: str):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor

def get_prediction(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return predicted_idx.item(), confidence.item()

def augment_image_variants(image_path: str, output_dir: str):
    """Generates 4 augmented variants of an image and returns their paths."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    variant_paths = []
    
    augment_fns = [
        ("rotate_scale_noise", augment_rotate_scale_noise),
        ("3d_warp_noise", augment_3d_warp_noise),
        ("rotate_partial", augment_rotate_partial),
        ("3d_partial", augment_3d_partial),
        ("rotate_scale_noisev2", augment_rotate_scale_noisev2),
        ("3d_warp_noisev2", augment_3d_warp_noisev2),
        ("rotate_partialv2", augment_rotate_partialv2),
        ("3d_partialv2", augment_3d_partialv2),
        ("colour_stressv2", augment_colour_stressv2)
    ]
 
    for suffix, fn in augment_fns:
        try:
            # The augmentation functions expect a path and return (image_bgr, rand_list, metadata)
            augmented_bgr, _, _ = fn(image_path)
            variant_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
            cv2.imwrite(variant_path, augmented_bgr)
            variant_paths.append(variant_path)
        except Exception as e:
            print(f"Error augmenting with {suffix}: {e}")

    return variant_paths

# Keep existing load_weights for compatibility if needed, but updated get_model is preferred
def load_weights(model, weight_path, strict=False):
    weight_dict = torch.load(weight_path, map_location='cpu')
    if 'model_state_dict' in weight_dict:
        weight_dict = weight_dict['model_state_dict']
    missing, unexpected = model.backbone.load_state_dict(weight_dict, strict=strict)
    return model
