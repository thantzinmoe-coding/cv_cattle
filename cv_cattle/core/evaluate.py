import json
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def evaluate_model():
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUTS_DIR / "models" / "cattle_pose_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained cattle model not found at {model_path}. Train the model before evaluation."
        )
        
    model = YOLO(str(model_path))
    
    # Evaluate strictly on test split
    metrics = model.val(data=str(PROJECT_ROOT / "dataset" / "data.yaml"), split="test", project=str(OUTPUTS_DIR / "yolo_runs"), name="test_eval", exist_ok=True)
    
    # Create metric payload
    metrics_dict = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "pose_map50": float(metrics.pose.map50),
        "pose_map50_95": float(metrics.pose.map)
    }
    
    # Write JSON output
    with open(OUTPUTS_DIR / "metrics" / "test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # Write CSV output
    df = pd.DataFrame([metrics_dict])
    df.to_csv(OUTPUTS_DIR / "metrics" / "test_metrics.csv", index=False)
    
    print("\nEvaluation Results on Test Set:")
    print("="*35)
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
    print("="*35)

if __name__ == "__main__":
    evaluate_model()
