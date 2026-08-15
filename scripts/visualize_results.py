import matplotlib.pyplot as plt
import cv2
import json
from pathlib import Path
from ultralytics import YOLO

def generate_comparisons():
    print("Generating validation batch comparisons...")
    
    out_dir = Path("outputs/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = "outputs/models/cattle_pose_best.pt"
    if not Path(model_path).exists():
        print(f"Falling back to default yolo pose")
        model_path = "weights/yolov8n-pose.pt"
        
    model = YOLO(model_path)
    
    # We test on validation set to generate the "val_batchX_pred.jpg" and "val_batchX_labels.jpg" comparisons
    metrics = model.val(data="dataset/data.yaml", split="val", save=True, save_dir="outputs/comparison", project="outputs", name="comparison", exist_ok=True)
    
    print("Generated Ground Truth vs Prediction Comparison Grids inside outputs/comparison/")
    
if __name__ == "__main__":
    generate_comparisons()
