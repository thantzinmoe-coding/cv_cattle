"""Clean the downloaded Roboflow cattle-detection dataset.

The source export is never modified. The script removes empty annotations,
normalizes polygon annotations to detection boxes, removes duplicate source
images, and rebuilds leakage-resistant train/valid/test splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "counts_cows_dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "counting_clean"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    base_name: str
    rows: tuple[str, ...]


def base_name(path: Path) -> str:
    """Remove the Roboflow hash while retaining the original source name."""
    return path.stem.split(".rf.", 1)[0]


def sequence_group(name: str) -> str:
    """Keep adjacent frames and duplicate source variants in one split."""
    value = re.sub(r"_(?:jpg|jpeg|png)$", "", name, flags=re.IGNORECASE)
    lowered = value.lower()
    match = re.fullmatch(r"(img|frame)\d+", lowered)
    if match:
        return match.group(1)
    match = re.fullmatch(r"((?:vid|hf)\d+)[-_]\d+", lowered)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{6,}", lowered):
        return "numeric_sequence"
    value = re.sub(r"[-_]\d+$", "", lowered)
    return value or lowered


def parse_label(path: Path) -> tuple[tuple[str, ...], int, int]:
    """Return detection rows plus converted-polygon and invalid-row counts."""
    rows: list[str] = []
    converted = 0
    invalid = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            invalid += 1
            continue

        if class_id != 0:
            invalid += 1
            continue

        if len(values) == 4:
            x, y, width, height = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            converted += 1
        else:
            invalid += 1
            continue

        if not all(0 <= value <= 1 for value in (x, y, width, height)):
            invalid += 1
            continue
        if width <= 0 or height <= 0:
            invalid += 1
            continue
        rows.append(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    return tuple(rows), converted, invalid


def pixel_hash(path: Path) -> str:
    """Hash decoded pixels so identical images with different JPEG metadata match."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}".encode())
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def assign_splits(records: list[Record], seed: int) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[sequence_group(record.base_name)].append(record)

    total = len(records)
    targets = {"train": total * 0.8, "valid": total * 0.1, "test": total * 0.1}
    result: dict[str, list[Record]] = {"train": [], "valid": [], "test": []}

    def group_order(item: tuple[str, list[Record]]) -> tuple[int, str]:
        group, members = item
        tie_break = hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()
        return (-len(members), tie_break)

    for _, members in sorted(groups.items(), key=group_order):
        split = max(
            result,
            key=lambda name: (targets[name] - len(result[name]), targets[name]),
        )
        result[split].extend(members)
    return result


def clean_dataset(source: Path, output: Path, seed: int, force: bool) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {output}. Use --force to rebuild it.")
        if output.resolve() == source.resolve() or PROJECT_ROOT not in output.resolve().parents:
            raise ValueError(f"Refusing to remove unsafe output path: {output}")
        shutil.rmtree(output)

    candidates: list[Record] = []
    stats: Counter[str] = Counter()
    for original_split in ("train", "valid", "test"):
        image_dir = source / original_split / "images"
        label_dir = source / original_split / "labels"
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                stats["missing_labels_removed"] += 1
                continue
            rows, converted, invalid = parse_label(label)
            stats["polygon_rows_converted"] += converted
            stats["invalid_rows_removed"] += invalid
            if not rows:
                stats["empty_images_removed"] += 1
                continue
            candidates.append(Record(image, label, base_name(image), rows))

    # Prefer one representative for repeated Roboflow exports of one source.
    by_base: dict[str, Record] = {}
    for record in candidates:
        key = record.base_name.lower()
        if key in by_base:
            stats["duplicate_source_names_removed"] += 1
            continue
        by_base[key] = record

    # Also catch identical pixels stored under different filenames.
    unique: list[Record] = []
    seen_hashes: set[str] = set()
    for record in by_base.values():
        digest = pixel_hash(record.image)
        if digest in seen_hashes:
            stats["duplicate_pixel_images_removed"] += 1
            continue
        seen_hashes.add(digest)
        unique.append(record)

    splits = assign_splits(unique, seed)
    object_counts: Counter[str] = Counter()
    for split, records in splits.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            shutil.copy2(record.image, image_dir / record.image.name)
            (label_dir / f"{record.image.stem}.txt").write_text(
                "\n".join(record.rows) + "\n", encoding="utf-8"
            )
            object_counts[split] += len(record.rows)

    (output / "data.yaml").write_text(
        "# Cleaned human-annotated cattle counting dataset\n"
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['cattle']\n",
        encoding="utf-8",
    )
    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "images": {name: len(records) for name, records in splits.items()},
        "objects": {name: object_counts[name] for name in splits},
        "cleaning": dict(stats),
        "split_policy": "80/10/10 grouped by source/video sequence",
        "seed": seed,
    }
    (output / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the Roboflow cattle dataset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(clean_dataset(args.source, args.output, args.seed, args.force), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
