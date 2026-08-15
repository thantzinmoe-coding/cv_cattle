import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}


@router.post("/image")
async def predict_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save uploaded file with a unique name to avoid collisions
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / unique_name
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run prediction using existing predict.py logic
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.predict import process_source
        process_source(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Find the most recently created output image in predictions dir
    result_files = sorted(PREDICTIONS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    result_image_url = None
    cow_count = None
    lameness_results = {}
    detected_ids = []

    # Try to get count from saved JSON
    counting_json = PROJECT_ROOT / "outputs" / "metrics" / "counting_results.json"
    if counting_json.exists():
        import json
        with open(counting_json) as jf:
            data = json.load(jf)
        cow_count = data.get("cows_detected") or data.get("unique_cows_detected")
        lameness_results = data.get("lameness_results", {})
        detected_ids = data.get("detected_ids", [])
        
        # If tracking loop explicitly output the video filename
        output_video_name = data.get("output_video")
        if output_video_name:
            result_image_url = f"/outputs/predictions/{output_video_name}"

    # Find the result image/video if not explicitly set
    if result_image_url is None:
        for f in result_files:
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                result_image_url = f"/outputs/predictions/{f.name}"
                break

    return JSONResponse({
        "message": "Prediction complete.",
        "result_image_url": result_image_url,
        "cow_count": cow_count,
        "original_filename": file.filename,
        "detected_ids": detected_ids,
        "lameness_results": lameness_results
    })

@router.post("/webcam")
async def start_webcam():
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.predict import process_source
        process_source("0")
        return {"message": "Webcam session finished natively"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webcam failed: {str(e)}")
