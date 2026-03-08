import redis
import time
import json
import os

from app.helpers import get_model, image_preprocess, get_prediction, augment_image_variants

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Cache loaded models
loaded_models = {}

def run_prediction(task_id: str, model_architecture: str, filename: str, file_path: str):
    try:
        redis_client.set(task_id, json.dumps({
            "status": "processing",
            "model_architecture": model_architecture,
            "filename": filename
        }))

        # Load model if not already loaded
        if model_architecture not in loaded_models:
            loaded_models[model_architecture] = get_model(model_architecture)
        
        model = loaded_models[model_architecture]
        
        # Preprocess image
        tensor = image_preprocess(file_path)
        
        # Get prediction
        predicted_idx, confidence = get_prediction(model, tensor)

        redis_client.set(task_id, json.dumps({
            "status": "completed",
            "result": f"Prediction for {filename} using {model_architecture} completed. Class ID: {predicted_idx}",
            "confidence": confidence,
            "predicted_idx": predicted_idx,
            "image_url": f"/uploads/{task_id}_{filename}"
        }))

    except Exception as e:
        redis_client.set(task_id, json.dumps({
            "status": "failed",
            "error": str(e)
        }))
    finally:
        # We handle cleanup manually or via a cron/ttl if needed, 
        # but for now we keep the image for the UI preview.
        pass

def run_augmentation(task_id: str, filename: str, file_path: str):
    try:
        redis_client.set(task_id, json.dumps({
            "status": "processing",
            "filename": filename
        }))

        # Generate variants in a separate directory within static/augmented
        # Resolve path relative to this file's location
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(BASE_DIR, "static", "augmented", task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        variant_paths = augment_image_variants(file_path, output_dir)
        
        # Convert absolute paths to relative URLs for the frontend
        variant_urls = [f"/static/augmented/{task_id}/{os.path.basename(p)}" for p in variant_paths]

        redis_client.set(task_id, json.dumps({
            "status": "completed",
            "filename": filename,
            "variants": variant_urls
        }))

    except Exception as e:
        redis_client.set(task_id, json.dumps({
            "status": "failed",
            "error": str(e)
        }))
    finally:
        # Keep the original for now (or move to a dedicated static path if preferred)
        pass
