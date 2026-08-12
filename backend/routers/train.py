import asyncio
import subprocess
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Global process handle so we can stream its output
_train_process: subprocess.Popen | None = None
_train_status: str = "idle"  # idle | running | done | error


class TrainConfig(BaseModel):
    epochs: int = 50
    imgsz: int = 640


@router.post("/start")
def start_training(config: TrainConfig):
    global _train_process, _train_status

    if _train_status == "running":
        raise HTTPException(status_code=409, detail="Training is already running.")

    # Patch train.py call so we can pass dynamic epochs/imgsz
    cmd = [
        sys.executable,
        "-c",
        f"""
import sys
sys.path.insert(0, r'{PROJECT_ROOT}')
from train import train_model
train_model(epochs={config.epochs}, imgsz={config.imgsz})
""",
    ]

    _train_process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _train_status = "running"
    return {"message": "Training started.", "epochs": config.epochs, "imgsz": config.imgsz}


@router.get("/status")
def get_status():
    global _train_process, _train_status

    if _train_process is not None:
        poll = _train_process.poll()
        if poll is None:
            _train_status = "running"
        elif poll == 0:
            _train_status = "done"
        else:
            _train_status = "error"

    model_exists = (PROJECT_ROOT / "outputs" / "models" / "cattle_pose_best.pt").exists()
    return {"status": _train_status, "trained_model_exists": model_exists}


@router.get("/logs")
def stream_logs():
    """Server-Sent Events stream of live training logs."""
    global _train_process

    def event_generator():
        if _train_process is None:
            yield "data: No training process started yet.\n\n"
            return

        for line in _train_process.stdout:
            yield f"data: {line.rstrip()}\n\n"

        # Signal completion
        return_code = _train_process.wait()
        if return_code == 0:
            yield "data: [TRAINING COMPLETE]\n\n"
        else:
            yield f"data: [ERROR] Process exited with code {return_code}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/stop")
def stop_training():
    global _train_process, _train_status
    if _train_process and _train_process.poll() is None:
        _train_process.terminate()
        _train_status = "idle"
        return {"message": "Training stopped."}
    return {"message": "No active training to stop."}
