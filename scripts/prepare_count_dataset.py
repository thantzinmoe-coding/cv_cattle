"""Prepare the annotation-free Roboflow cattle images for detector training.

The supplied export contains images and empty label files. This script uses a
COCO-pretrained detector to create *pseudo-labels* for the ``cow`` class, then
creates deterministic train/val/test splits under ``dataset/counting``.
Original files in ``dataset/counts_cows_dataset`` are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "counts_cows_dataset" / "train" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "counting"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_COW_CLASS = 19


def source_group(path: Path) -> str:
    """Group augmented copies and adjacent video frames into one split."""
    stem = path.stem.split(".rf.", 1)[0]
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:[-_]\d+)$", "", stem)
    return stem or path.stem


def split_for(path: Path, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{source_group(path)}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def prepare(
    source: Path,
    output: Path,
    model_path: str,
    confidence: float,
    seed: int,
    batch_size: int,
) -> dict:
    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No images found in {source}")

    for split in ("train", "val", "test"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    split_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    skipped = []

    def iter_results():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            batch_results = model.predict(
                source=[str(path) for path in batch],
                classes=[COCO_COW_CLASS],
                conf=confidence,
                iou=0.6,
                device=0,
                batch=batch_size,
                verbose=False,
            )
            # For list input Ultralytics assigns synthetic result paths such
            # as image0.jpg, so retain the real source path explicitly.
            yield from zip(batch, batch_results)
            done = min(start + batch_size, len(images))
            if done % 160 == 0 or done == len(images):
                print(f"Pseudo-labelled {done}/{len(images)} images", flush=True)

    for source_path, result in iter_results():
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            # These are not trustworthy background examples: the export is
            # advertised as cattle imagery, so a miss may be a teacher error.
            skipped.append(source_path.name)
            continue

        split = split_for(source_path, seed)
        destination_image = output / split / "images" / source_path.name
        destination_label = output / split / "labels" / f"{source_path.stem}.txt"
        shutil.copy2(source_path, destination_image)

        rows = []
        for x_center, y_center, width, height in boxes.xywhn.cpu().tolist():
            rows.append(
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
        destination_label.write_text("\n".join(rows) + "\n", encoding="utf-8")
        split_counts[split] += 1
        object_counts[split] += len(rows)

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        "# Pseudo-labelled cattle counting dataset\n"
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['cow']\n",
        encoding="utf-8",
    )

    report = {
        "source_images": len(images),
        "pseudo_label_confidence": confidence,
        "teacher_model": model_path,
        "images": {split: split_counts[split] for split in ("train", "val", "test")},
        "objects": {split: object_counts[split] for split in ("train", "val", "test")},
        "skipped_without_detection": len(skipped),
        "skipped_examples": skipped[:100],
        "warning": "Labels are model-generated and are not human ground truth.",
    }
    report_path = output / "pseudo_label_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Pseudo-label the cattle counting dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=str(PROJECT_ROOT / "yolov8n.pt"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = prepare(
        args.source, args.output, args.model, args.confidence, args.seed, args.batch_size
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
