import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mimetypes
from fastapi.staticfiles import StaticFiles

# Force strict video mimetypes for Windows environments
mimetypes.init()
mimetypes.add_type("video/webm", ".webm")
mimetypes.add_type("video/mp4", ".mp4")

# Ensure project root is on the path so prediction modules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import predict

app = FastAPI(title="Cattle Counting API", version="1.0.0")

# Allow requests from the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount output artifacts as static files (predicted images, plots, etc.)
outputs_dir = Path(__file__).parent.parent / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_dir)), name="outputs")

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])


@app.get("/health")
def health_check():
    model_path = Path(__file__).parent.parent / "outputs" / "models" / "cattle_count_best.pt"
    return {
        "status": "ok",
        "model_available": model_path.exists(),
        "trained_model_exists": model_path.exists(),
    }
