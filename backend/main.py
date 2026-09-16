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

# Automatically inject the bundled FFmpeg binary path for videos
try:
    import imageio_ffmpeg
    os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

# Ensure project root is on the path so prediction modules can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import predict

app = FastAPI(title="Cattle Counting API", version="1.0.0")

# Allow requests from the Vite dev server on any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
