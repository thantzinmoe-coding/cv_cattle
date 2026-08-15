import json
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys
import uuid
import imageio
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(PROJECT_ROOT))
from core.lameness_classifier import LamenessPredictor

def normalize_keypoints(keypoints, box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0: return keypoints
    norm_kp = np.zeros_like(keypoints)
    norm_kp[:, 0] = (keypoints[:, 0] - x1) / w
    norm_kp[:, 1] = (keypoints[:, 1] - y1) / h
    return norm_kp

def process_source(source_path):
    (OUTPUTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUTS_DIR / "models" / "cattle_pose_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained cattle model not found at {model_path}.")
        
    model = YOLO(str(model_path))
    lameness_model_path = str(OUTPUTS_DIR / "models" / "lameness_model.pkl")
    lameness_predictor = LamenessPredictor(lameness_model_path)
    
    is_webcam = str(source_path) == "0"
    is_video = str(source_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) or is_webcam
    
    if is_video:
        source_val = 0 if is_webcam else str(source_path)
        print(f"Running video tracking algorithm on {source_val}...")
        
        cap = cv2.VideoCapture(source_val)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        
        if not is_webcam:
            unique_name = f"{uuid.uuid4().hex}.mp4"
            out_video_path = OUTPUTS_DIR / "predictions" / unique_name
            
            # Use imageio with standard libx264 to guarantee browser compatibility
            writer = imageio.get_writer(str(out_video_path), fps=fps, codec='libx264', macro_block_size=None, format='FFMPEG')
        
        track_history = {}
        lameness_results = {}
        unique_cow_ids = set()
        max_untracked_cows = 0
        
        results_gen = model.track(source=source_val, tracker="botsort.yaml", stream=True, persist=True)
        
        for r in results_gen:
            frame = r.plot()
            if r.boxes is not None:
                max_untracked_cows = max(max_untracked_cows, len(r.boxes))
            
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().numpy()
                kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
                
                unique_cow_ids.update(ids)
                
                for i, cow_id in enumerate(ids):
                    if kpts is not None and len(kpts[i]) > 0:
                        norm_kpt = normalize_keypoints(kpts[i], boxes[i])
                        if cow_id not in track_history:
                            track_history[cow_id] = []
                        track_history[cow_id].append(norm_kpt)
                        
                        if len(track_history[cow_id]) >= 30:
                            status, conf = lameness_predictor.predict_lameness(track_history[cow_id])
                            lameness_results[cow_id] = {"status": status, "confidence": float(conf)}
                            
                            x1, y1, x2, y2 = map(int, boxes[i])
                            text = f"{status} {int(conf*100)}%"
                            color = (0, 0, 255) if status == "POSSIBLE LAMENESS" else (0, 255, 0)
                            cv2.putText(frame, text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if not is_webcam:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame_rgb)
            else:
                cv2.imshow("Cattle Lameness Monitor (Press Q to exit webcam)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()
        if not is_webcam:
            writer.close()
        cv2.destroyAllWindows()
        
        cow_count = len(unique_cow_ids) if len(unique_cow_ids) > 0 else max_untracked_cows
        
        counting_results = {
            "source": str(source_path),
            "unique_cows_detected": cow_count,
            "detected_ids": list(unique_cow_ids),
            "lameness_results": lameness_results,
            "output_video": unique_name
        }
    else:
        print(f"Running static prediction on {source_path}...")
        results = model.predict(source=str(source_path), save=True, project=str(OUTPUTS_DIR), name="predictions", exist_ok=True)
        
        total_detections = 0
        for r in results:
            total_detections += len(r.boxes)
            
        counting_results = {
            "source": str(source_path),
            "cows_detected": total_detections,
            "lameness_results": {} 
        }
        
    with open(OUTPUTS_DIR / "metrics" / "counting_results.json", "w") as f:
        json.dump(counting_results, f, indent=4)
    
    print("\nVisual representations saved dynamically to outputs/predictions/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cattle Pose Prediction and Object Tracking.")
    parser.add_argument("--source", required=True, help="Path to image, video or folder of images.")
    args = parser.parse_args()
    process_source(args.source)
