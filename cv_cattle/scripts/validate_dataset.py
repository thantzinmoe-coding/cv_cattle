"""Validate a YOLO pose dataset before training.

Checks image/label pairing, record width, normalized coordinates, visibility
values, readable images, and filename leakage between dataset splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def add_issue(report: dict, severity: str, code: str, path: Path, detail: str) -> None:
    report[severity].append({"code": code, "path": str(path), "detail": detail})


def validate_dataset(dataset_dir: Path, keypoint_count: int = 12) -> dict:
    expected_fields = 5 + keypoint_count * 3
    report = {
        "dataset": str(dataset_dir.resolve()),
        "keypoint_count": keypoint_count,
        "expected_fields_per_object": expected_fields,
        "splits": {},
        "visibility_counts": {},
        "errors": [],
        "warnings": [],
    }
    stems_by_split: dict[str, set[str]] = {}
    visibility_counts: Counter[int] = Counter()

    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        } if image_dir.exists() else {}
        labels = {path.stem: path for path in label_dir.glob("*.txt")} if label_dir.exists() else {}
        stems_by_split[split] = set(images)
        object_count = 0

        for stem in sorted(set(images) - set(labels)):
            add_issue(report, "errors", "missing_label", images[stem], "No matching label file")
        for stem in sorted(set(labels) - set(images)):
            add_issue(report, "errors", "missing_image", labels[stem], "No matching image file")

        for image_path in images.values():
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                add_issue(report, "errors", "unreadable_image", image_path, str(exc))

        for label_path in labels.values():
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not raw_line.strip():
                    continue
                object_count += 1
                fields = raw_line.split()
                if len(fields) != expected_fields:
                    add_issue(
                        report,
                        "errors",
                        "wrong_field_count",
                        label_path,
                        f"line {line_number}: expected {expected_fields}, found {len(fields)}",
                    )
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    add_issue(report, "errors", "non_numeric_label", label_path, f"line {line_number}: {exc}")
                    continue

                if class_id != 0:
                    add_issue(report, "errors", "invalid_class", label_path, f"line {line_number}: {class_id}")

                bbox = values[:4]
                if any(value < 0 or value > 1 for value in bbox):
                    add_issue(report, "errors", "bbox_out_of_range", label_path, f"line {line_number}: {bbox}")
                if bbox[2] <= 0 or bbox[3] <= 0:
                    add_issue(report, "errors", "invalid_bbox_size", label_path, f"line {line_number}: {bbox[2:]}")

                keypoints = values[4:]
                for index in range(keypoint_count):
                    x, y, visibility_raw = keypoints[index * 3:index * 3 + 3]
                    if not visibility_raw.is_integer() or int(visibility_raw) not in {0, 1, 2}:
                        add_issue(
                            report,
                            "errors",
                            "invalid_visibility",
                            label_path,
                            f"line {line_number}, keypoint {index}: {visibility_raw}",
                        )
                        continue
                    visibility = int(visibility_raw)
                    visibility_counts[visibility] += 1
                    if visibility > 0 and (x < 0 or x > 1 or y < 0 or y > 1):
                        add_issue(
                            report,
                            "errors",
                            "keypoint_out_of_range",
                            label_path,
                            f"line {line_number}, keypoint {index}: ({x}, {y})",
                        )

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": object_count,
        }

    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = sorted(stems_by_split[left] & stems_by_split[right])
            for stem in overlap:
                add_issue(
                    report,
                    "errors",
                    "split_leakage",
                    dataset_dir,
                    f"{stem!r} occurs in both {left} and {right}",
                )

    report["visibility_counts"] = {str(key): visibility_counts[key] for key in sorted(visibility_counts)}
    report["valid"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YOLO pose dataset integrity")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--keypoints", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/dataset_audit.json"))
    args = parser.parse_args()

    report = validate_dataset(args.dataset, args.keypoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "valid": report["valid"],
        "splits": report["splits"],
        "visibility_counts": report["visibility_counts"],
        "errors": len(report["errors"]),
        "warnings": len(report["warnings"]),
        "report": str(args.output),
    }, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
