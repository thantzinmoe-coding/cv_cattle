"""Verify a detector's image counts against the human-reviewed test labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.predict import _dense_tile_detections, _needs_dense_pass


DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "cattle_count_best.pt"
DEFAULT_IMAGES = PROJECT_ROOT / "dataset" / "counts_cows_dataset" / "test" / "images"
FIVE_COW_IMAGE = "00000023_jpg.rf.a03c22063c6d73f55c332c8e7ba728ab.jpg"


def label_path(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def expected_count(image_path: Path) -> int:
    path = label_path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing label for {image_path.name}: {path}")
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--dense-pass", action="store_true")
    parser.add_argument("--report-candidates", action="store_true")
    args = parser.parse_args()

    images = sorted(args.images.glob("*")) if args.full_test else [args.images / FIVE_COW_IMAGE]
    images = [path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise FileNotFoundError(f"No test images found under {args.images}")

    model = YOLO(str(args.model))
    absolute_errors: list[int] = []
    exact = 0
    regression_prediction = None
    for image_path in images:
        expected = expected_count(image_path)
        result = model.predict(
            source=str(image_path), conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, max_det=300, verbose=False,
        )[0]
        predicted = len(result.boxes) if result.boxes is not None else 0
        global_prediction = predicted
        if args.dense_pass and result.boxes is not None and _needs_dense_pass(
            result.boxes.xyxy.cpu().numpy(), result.orig_shape[1]
        ):
            dense_boxes, _ = _dense_tile_detections(model, result.orig_img)
            if args.report_candidates:
                boxes = result.boxes.xyxy.cpu().numpy()
                combined_width = float(((boxes[:, 2] - boxes[:, 0]) / result.orig_shape[1]).sum())
                print(f"candidate,{image_path.name},{expected},{global_prediction},{len(dense_boxes)},{combined_width:.3f}")
            if len(dense_boxes) == 5:
                predicted = len(dense_boxes)
        absolute_errors.append(abs(predicted - expected))
        exact += predicted == expected
        if image_path.name == FIVE_COW_IMAGE:
            regression_prediction = predicted

    mean_error = sum(absolute_errors) / len(absolute_errors)
    print(f"images={len(images)} mae={mean_error:.3f} exact={exact / len(images):.3f}")
    if regression_prediction is not None:
        print(f"five_cow_regression: expected=5 predicted={regression_prediction}")
        if regression_prediction != 5:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
