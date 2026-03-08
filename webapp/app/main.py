from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uuid
import json
import os
from app.worker import run_prediction, run_augmentation, redis_client
from app.helpers import MODELS_CONFIG

app = FastAPI(title="DL Model WebApp API")

# Path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # webapp/app
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(os.path.dirname(BASE_DIR), "uploads")

# Ensure directories exist for mounting
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/models")
async def get_models():
    return list(MODELS_CONFIG.keys())

@app.get("/class-mapping")
async def get_class_mapping():
    return FileResponse(os.path.join(STATIC_DIR, "resnet34_v1_class_mapping.json"))

# Serve uploaded images for preview
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.post("/predict")
async def create_prediction(
    background_tasks: BackgroundTasks,
    model_architecture: str = Form(...),
    image: UploadFile = File(...)
):
    task_id = str(uuid.uuid4())
    filename = image.filename
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    redis_client.set(task_id, json.dumps({
        "status": "queued",
        "model_architecture": model_architecture,
        "filename": filename,
        "file_path": file_path
    }))
 
    background_tasks.add_task(run_prediction, task_id, model_architecture, filename, file_path)

    return {"message": "Prediction task started", "task_id": task_id}

@app.post("/augment")
async def create_augmentation(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...)
):
    task_id = str(uuid.uuid4())
    filename = image.filename
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}_{filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    redis_client.set(task_id, json.dumps({
        "status": "queued",
        "filename": filename,
        "file_path": file_path
    }))
 
    background_tasks.add_task(run_augmentation, task_id, filename, file_path)

    return {"message": "Augmentation task started", "task_id": task_id}

@app.get("/predict-result/{task_id}")
async def get_predict_result(task_id: str):
    result = redis_client.get(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task ID not found")

    return json.loads(result)

@app.post("/feature-map")
async def visualize_feature_map(image: UploadFile = File(...)):
    return {"message": f"Feature map visualization for {image.filename} (Not Implemented)", "filename": image.filename}

