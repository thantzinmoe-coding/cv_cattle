import json
from ultralytics import YOLO
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def process_source(source_path):
    (OUTPUTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUTS_DIR / "models" / "cattle_pose_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained cattle model not found at {model_path}. Train the model before prediction."
        )
        
    model = YOLO(str(model_path))
    
    is_video = str(source_path).lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # Ensure source exists (if it's a path)
    if isinstance(source_path, str) and not Path(source_path).exists():
        try:
            # Let YOLO handle it (could be YouTube URL or other stream)
            pass
        except:
            print(f"Error: Could not locate source file at {source_path}")
    
    if is_video:
        print(f"Running video tracking algorithm on {source_path}...")
        # ByteTrack is enabled natively via tracker="bytetrack.yaml"
        results = model.track(source=source_path, tracker="bytetrack.yaml", save=True, project=str(OUTPUTS_DIR), name="predictions", exist_ok=True, device=0)
        unique_cow_ids = set()
        
        for r in results:
            if r.boxes.id is not None:
                # Add all tracked IDs in this frame
                ids = r.boxes.id.int().cpu().tolist()
                unique_cow_ids.update(ids)
        
        cow_count = len(unique_cow_ids)
        print("\n" + "="*30)
        for cow_id in sorted(list(unique_cow_ids)):
            print(f"Cow ID: {cow_id}")
        print(f"\nTotal cows: {cow_count}")
        print("="*30)
        
        counting_results = {
            "source": source_path,
            "unique_cows_detected": cow_count,
            "detected_ids": list(unique_cow_ids)
        }
    else:
        print(f"Running static prediction on {source_path}...")
        results = model.predict(source=source_path, save=True, project=str(OUTPUTS_DIR), name="predictions", exist_ok=True, device=0)
        
        total_detections = 0
        for r in results:
            total_detections += len(r.boxes)
            
        print("\n" + "="*30)
        print(f"Detected cows: {total_detections}")
        print("="*30)
        
        counting_results = {
            "source": source_path,
            "cows_detected": total_detections
        }
        
    with open(OUTPUTS_DIR / "metrics" / "counting_results.json", "w") as f:
        json.dump(counting_results, f, indent=4)
    
    print("\nVisual representations with 12-keypoint skeletons saved dynamically to outputs/predictions/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cattle Pose Prediction and Object Tracking.")
    parser.add_argument("--source", required=True, help="Path to image, video or folder of images.")
    args = parser.parse_args()
    process_source(args.source)
