import json
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_YAML = PROJECT_ROOT / "dataset" / "counting_clean" / "data.yaml"

def evaluate_model():
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    
    model_path = OUTPUTS_DIR / "models" / "cattle_count_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained cattle model not found at {model_path}. Train the model before evaluation."
        )
        
    model = YOLO(str(model_path))
    
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Counting dataset not found at {DATASET_YAML}. "
            "Run: python scripts/clean_count_dataset.py"
        )

    metrics = model.val(
        data=str(DATASET_YAML),
        split="test",
        project=str(OUTPUTS_DIR / "yolo_runs"),
        name="count_test_eval",
        exist_ok=True,
    )
    
    # Create metric payload
    metrics_dict = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "images": len(list((PROJECT_ROOT / "dataset" / "counting_clean" / "test" / "images").glob("*"))),
        "labels": "human-reviewed"
    }
    
    # Write JSON output
    with open(OUTPUTS_DIR / "metrics" / "count_test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # Write CSV output
    df = pd.DataFrame([metrics_dict])
    df.to_csv(OUTPUTS_DIR / "metrics" / "count_test_metrics.csv", index=False)
    
    print("\nEvaluation Results on Test Set:")
    print("="*35)
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print("="*35)

if __name__ == "__main__":
    evaluate_model()
