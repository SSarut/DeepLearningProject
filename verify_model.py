import torch
from webapp.app.helpers import get_model, image_preprocess, get_prediction
from PIL import Image
import os

def test_prediction():
    print("Testing model loading and prediction...")
    
    # 1. Create a dummy image
    img_path = "test_image.jpg"
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img.save(img_path)
    print(f"Dummy image created at {img_path}")
    
    try:
        # 2. Get model
        model_name = "resnet34"
        print(f"Loading model: {model_name}")
        model = get_model(model_name)
        print("Model loaded successfully")
        
        # 3. Preprocess
        print("Preprocessing image...")
        tensor = image_preprocess(img_path)
        print(f"Tensor shape: {tensor.shape}")
        
        # 4. Predict
        print("Running prediction...")
        predicted_idx, confidence = get_prediction(model, tensor)
        print(f"Prediction: Class ID {predicted_idx}, Confidence {confidence:.4f}")
        
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == "__main__":
    test_prediction()
