import json
import threading
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
METRICS_FILE = PROJECT_ROOT / "outputs" / "metrics" / "count_test_metrics.json"

_eval_status: str = "idle"  # idle | running | done | error
_eval_error: str = ""


def _run_evaluation():
    global _eval_status, _eval_error
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.evaluate import evaluate_model
        evaluate_model()
        _eval_status = "done"
    except Exception as e:
        _eval_error = str(e)
        _eval_status = "error"


@router.post("/run")
def run_evaluation():
    global _eval_status, _eval_error

    if _eval_status == "running":
        raise HTTPException(status_code=409, detail="Evaluation is already running.")

    _eval_status = "running"
    _eval_error = ""
    thread = threading.Thread(target=_run_evaluation, daemon=True)
    thread.start()

    return {"message": "Evaluation started in background."}


@router.get("/status")
def get_eval_status():
    return {"status": _eval_status, "error": _eval_error}


@router.get("/metrics")
def get_metrics():
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation metrics found. Run evaluation first.",
        )

    with open(METRICS_FILE) as f:
        data = json.load(f)

    return JSONResponse(data)
