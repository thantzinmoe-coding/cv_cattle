import uuid
import base64
import asyncio
import threading
from pathlib import Path
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np

router = APIRouter()

PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_DIR      = PROJECT_ROOT / "outputs" / "uploads"
PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Lazily loaded model shared across WebSocket connections
_ws_model = None
_ws_model_lock = threading.Lock()
_ws_inference_lock = threading.Lock()
_upload_inference_lock = asyncio.Lock()


def _get_ws_model():
    """Load the YOLO model once and reuse it for all WebSocket frames."""
    global _ws_model
    with _ws_model_lock:
        if _ws_model is None:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT))
            from ultralytics import YOLO
            model_path = PROJECT_ROOT / "outputs" / "models" / "cattle_count_best.pt"
            if not model_path.exists():
                raise FileNotFoundError("Cattle detection model is not installed on the server.")
            _ws_model = YOLO(str(model_path))
        return _ws_model


@router.post("/image")
async def predict_image(
    file: UploadFile = File(...),
    line_orientation: str = Form("vertical"),
    line_position: float = Form(0.5),
    roi_x1: float = Form(0.0),
    roi_y1: float = Form(0.0),
    roi_x2: float = Form(1.0),
    roi_y2: float = Form(1.0),
    confidence: float = Form(0.30),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    job_id = uuid.uuid4().hex
    unique_name = f"{job_id}{suffix}"
    save_path = UPLOAD_DIR / unique_name
    received = 0
    try:
        with open(save_path, "wb") as destination:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds the 500 MB limit.")
                destination.write(chunk)
    except Exception:
        save_path.unlink(missing_ok=True)
        raise

    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from core.predict import CountingConfig, process_source
        config = CountingConfig(
            orientation=line_orientation,
            line_position=line_position,
            roi_x1=roi_x1,
            roi_y1=roi_y1,
            roi_x2=roi_x2,
            roi_y2=roi_y2,
            confidence=confidence,
        )
        async with _upload_inference_lock:
            result = await asyncio.to_thread(process_source, str(save_path), config, job_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    output_name = result.get("output_video") or result.get("output_image")
    result_url = f"/outputs/predictions/{output_name}" if output_name else None
    cow_count = result.get("cows_detected", result.get("unique_cows_detected", 0))

    return JSONResponse({
        "message": "Prediction complete.",
        "job_id": job_id,
        "result_image_url": result_url,
        "cow_count": cow_count,
        "count_mode": result.get("count_mode"),
        "forward_count": result.get("forward_count", 0),
        "reverse_count": result.get("reverse_count", 0),
        "peak_visible_count": result.get("peak_visible_count", cow_count),
        "original_filename": file.filename,
        "detected_ids": result.get("detected_ids", []),
        "lameness_results": result.get("lameness_results", {}),
        "condition_results": result.get("condition_results", {}),
        "counting_config": result.get("counting_config", {}),
    })


@router.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket):
    """
    WebSocket endpoint for real-time webcam cow detection.

    Protocol:
      Client → Server : raw JPEG bytes of a single camera frame
      Server → Client : JSON { cow_count, annotated_frame_b64 }

    The frontend captures frames from the browser camera using getUserMedia +
    Canvas, and sends them here for inference. This avoids opening any native
    OpenCV window on the server side.
    """
    await websocket.accept()

    try:
        model = _get_ws_model()
    except FileNotFoundError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from core.predict import (
        IOU_THRESHOLD,
        CountingConfig,
        LineCrossingCounter,
        assess_track_condition,
        draw_counting_geometry,
    )

    try:
        config = CountingConfig(
            orientation=websocket.query_params.get("orientation", "vertical"),
            line_position=float(websocket.query_params.get("position", "0.5")),
            roi_x1=float(websocket.query_params.get("roi_x1", "0")),
            roi_y1=float(websocket.query_params.get("roi_y1", "0")),
            roi_x2=float(websocket.query_params.get("roi_x2", "1")),
            roi_y2=float(websocket.query_params.get("roi_y2", "1")),
            confidence=float(websocket.query_params.get("confidence", "0.3")),
        )
    except ValueError as error:
        await websocket.send_json({"error": str(error)})
        await websocket.close(code=1008)
        return

    loop = asyncio.get_event_loop()
    track_history = {}
    confidence_history = {}
    crossing_counter = LineCrossingCounter(config)

    try:
        while True:
            # Receive raw JPEG bytes from the browser
            jpeg_bytes = await websocket.receive_bytes()

            # Decode to numpy array
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Run YOLO predict on this single frame (non-blocking via executor)
            def _infer(frm):
                with _ws_inference_lock:
                    results = model.track(
                        source=frm,
                        tracker="bytetrack.yaml",
                        persist=True,
                        conf=config.confidence,
                        iou=IOU_THRESHOLD,
                        verbose=False,
                    )
                r = results[0]
                visible_count = 0
                annotated = r.plot()
                conditions = []

                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().tolist()
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confidences = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [0.0] * len(ids)
                    frame_height, frame_width = r.orig_shape
                    for index, cow_id in enumerate(ids):
                        x1, y1, x2, y2 = boxes[index]
                        center = [((x1 + x2) / 2) / frame_width, ((y1 + y2) / 2) / frame_height]
                        if not config.contains(center):
                            continue
                        visible_count += 1
                        history = track_history.setdefault(cow_id, [])
                        history.append(center)
                        if len(history) > 60:
                            del history[:-60]
                        confidence_values = confidence_history.setdefault(cow_id, [])
                        confidence_values.append(float(confidences[index]))
                        if len(confidence_values) > 60:
                            del confidence_values[:-60]
                        conditions.append({
                            "cow_id": cow_id,
                            **assess_track_condition(history, confidence_values),
                        })

                        crossing_counter.update(cow_id, center)

                draw_counting_geometry(annotated, config)
                overlay_text = (
                    f"Crossed: {crossing_counter.total}  "
                    f"Forward: {crossing_counter.forward_count}  "
                    f"Reverse: {crossing_counter.reverse_count}  "
                    f"Visible: {visible_count}"
                )
                (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(annotated, (8, 8), (16 + tw, 20 + th + 8), (0, 0, 0), -1)
                cv2.putText(annotated, overlay_text, (12, 20 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 180), 2, cv2.LINE_AA)

                # Encode annotated frame as JPEG for sending back
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return visible_count, conditions, base64.b64encode(buf.tobytes()).decode()

            visible_count, conditions, frame_b64 = await loop.run_in_executor(None, _infer, frame)

            await websocket.send_json({
                "cow_count": crossing_counter.total,
                "visible_count": visible_count,
                "forward_count": crossing_counter.forward_count,
                "reverse_count": crossing_counter.reverse_count,
                "conditions": conditions,
                "annotated_frame_b64": frame_b64,
            })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@router.post("/webcam")
async def start_webcam():
    """Legacy endpoint — kept for compatibility. Real-time webcam now uses WebSocket."""
    return {"message": "Use the WebSocket endpoint /predict/ws/webcam for live webcam detection."}
