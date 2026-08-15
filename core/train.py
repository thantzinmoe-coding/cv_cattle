import os
import shutil
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from scripts.validate_dataset import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASET_YAML = PROJECT_ROOT / "dataset" / "processed" / "data.yaml"

def setup_directories():
    for d in ['outputs/models', 'outputs/metrics', 'outputs/plots']:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

def train_model(epochs=50, imgsz=640):
    setup_directories()

    audit = validate_dataset(PROJECT_ROOT / "dataset", keypoint_count=12)
    if not audit["valid"]:
        raise RuntimeError(
            f"Dataset validation failed with {len(audit['errors'])} error(s). "
            "Run: python scripts/validate_dataset.py"
        )
    
    print("\n[3/3] Training YOLOv8-pose model...")
    # Initialize YOLOv8 Pose model using the nano weights for quick deployment/transfer
    model = YOLO(str(PROJECT_ROOT / "weights" / "yolov8n-pose.pt"))
    
    # Run training
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        project=str(OUTPUTS_DIR / "yolo_runs"),
        name="train",
        exist_ok=True,
        patience=15,
        workers=4,
        seed=42,
        deterministic=True,
    )
    
    # Copy best model to outputs/models
    best_model_path = OUTPUTS_DIR / "yolo_runs" / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        shutil.copy(best_model_path, OUTPUTS_DIR / "models" / "cattle_pose_best.pt")
        print("Model saved to outputs/models/cattle_pose_best.pt")
        
    # Copy metrics to outputs/metrics
    results_csv_path = OUTPUTS_DIR / "yolo_runs" / "train" / "results.csv"
    if results_csv_path.exists():
        shutil.copy(results_csv_path, OUTPUTS_DIR / "metrics" / "training_metrics.csv")
        print("Training metrics saved to outputs/metrics/training_metrics.csv")
        
    # Copy all graphs and plots to outputs/plots
    yolo_run_dir = OUTPUTS_DIR / "yolo_runs" / "train"
    for plot_file in yolo_run_dir.glob("*.png"):
        shutil.copy(plot_file, OUTPUTS_DIR / "plots" / plot_file.name)
        
    print("Training Complete. All artifacts exported successfully.")

if __name__ == "__main__":
    train_model()
