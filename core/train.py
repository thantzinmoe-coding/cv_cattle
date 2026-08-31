import shutil
import argparse
from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DATASET_YAML = PROJECT_ROOT / "dataset" / "counting_clean" / "data.yaml"


def setup_directories():
    for d in ["outputs/models", "outputs/metrics", "outputs/plots"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


def train_model(epochs=50, imgsz=640, batch=4):
    setup_directories()

    # Verify processed dataset exists
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Counting dataset not found at {DATASET_YAML}. "
            "Run: python scripts/clean_count_dataset.py"
        )

    print("\n[1/3] Dataset verified.")
    print(f"      YAML: {DATASET_YAML}")

    # YOLO11 detection (not pose — the counting dataset has boxes only).
    # Use ultralytics auto-download if the local weight isn't present.
    local_weights = [
        PROJECT_ROOT / "yolo11s.pt",
        PROJECT_ROOT / "weights" / "yolo11s.pt",
    ]
    local_weight = next((path for path in local_weights if path.exists()), None)
    model_weight = str(local_weight) if local_weight else "yolo11s.pt"
    print(f"\n[2/3] Loading base model: {model_weight}")
    model = YOLO(model_weight)

    print(f"\n[3/3] Training YOLO11s detection model for {epochs} epochs ...")
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(OUTPUTS_DIR / "yolo_runs"),
        name="count_train",
        exist_ok=True,
        patience=15,
        workers=4,
        seed=42,
        deterministic=True,
        # Moderate augmentation helps across aerial, side, and walkway views.
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.1,
        mosaic=1.0,
        mixup=0.1,
        degrees=10.0,
        scale=0.5,
        translate=0.1,
    )

    # Copy best weights to outputs/models
    best = OUTPUTS_DIR / "yolo_runs" / "count_train" / "weights" / "best.pt"
    if best.exists():
        dest = OUTPUTS_DIR / "models" / "cattle_count_best.pt"
        shutil.copy(best, dest)
        print(f"\nModel saved → {dest}")

    # Copy training metrics CSV
    results_csv = OUTPUTS_DIR / "yolo_runs" / "count_train" / "results.csv"
    if results_csv.exists():
        shutil.copy(results_csv, OUTPUTS_DIR / "metrics" / "count_training_metrics.csv")
        print("Training metrics saved -> outputs/metrics/count_training_metrics.csv")

    # Copy plots
    yolo_run_dir = OUTPUTS_DIR / "yolo_runs" / "count_train"
    count_plots_dir = OUTPUTS_DIR / "plots" / "counting"
    count_plots_dir.mkdir(parents=True, exist_ok=True)
    for plot_file in yolo_run_dir.glob("*.png"):
        shutil.copy(plot_file, count_plots_dir / plot_file.name)

    print("\nTraining complete. All artifacts exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the cattle counting detector.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
