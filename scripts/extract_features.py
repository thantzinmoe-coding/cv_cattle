"""Extract cattle motion features from labeled raw lameness videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.lameness_classifier import (  # noqa: E402
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SEQUENCE_LENGTH,
    MOTION_FEATURE_VERSION,
    extract_motion_features,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "raw" / "CattleLameness" / "Data"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "cattle_count_best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_features"


def _video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if np.isfinite(fps) and fps > 0 else 30.0


def _continuous_segments(samples, max_gap_frames: int):
    """Split a tracker history where the detector lost the animal too long."""
    if not samples:
        return []
    segments = [[samples[0]]]
    for sample in samples[1:]:
        if sample[0] - segments[-1][-1][0] > max_gap_frames:
            segments.append([])
        segments[-1].append(sample)
    return segments


def _resample_segment(segment, fps: float, sample_fps: float) -> np.ndarray:
    frames = np.asarray([sample[0] for sample in segment], dtype=np.float64)
    points = np.asarray([sample[1] for sample in segment], dtype=np.float64)
    step = fps / sample_fps
    targets = np.arange(frames[0], frames[-1] + 1e-6, step)
    if len(targets) < 3:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack([
        np.interp(targets, frames, points[:, 0]),
        np.interp(targets, frames, points[:, 1]),
    ])


def _sequence_windows(points: np.ndarray, sequence_length: int, stride: int):
    if len(points) < sequence_length:
        return []
    starts = list(range(0, len(points) - sequence_length + 1, stride))
    final_start = len(points) - sequence_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [points[start:start + sequence_length] for start in starts]


def extract_video_features(
    video_path: Path,
    model,
    sequence_length: int,
    sample_fps: float,
    confidence: float,
    image_size: int,
):
    """Extract windows from the longest continuous cattle track in a video."""
    fps = _video_fps(video_path)
    tracks: dict[int, list[tuple[int, list[float]]]] = {}
    frame_index = 0
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        stream=True,
        persist=False,
        conf=confidence,
        iou=0.5,
        imgsz=image_size,
        verbose=False,
    )
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()
            frame_height, frame_width = result.orig_shape
            for index, track_id in enumerate(ids):
                x1, y1, x2, y2 = boxes[index]
                center = [
                    float(((x1 + x2) / 2) / frame_width),
                    float(((y1 + y2) / 2) / frame_height),
                ]
                tracks.setdefault(track_id, []).append((frame_index, center))
        frame_index += 1

    max_gap_frames = max(2, round(fps * 0.35))
    segments = [
        segment
        for samples in tracks.values()
        for segment in _continuous_segments(samples, max_gap_frames)
    ]
    if not segments:
        return [], {"frames": frame_index, "reason": "no tracked cattle"}

    # The folder label describes the featured animal. Using only the longest
    # track avoids assigning that label to short background detections.
    dominant = max(segments, key=lambda segment: segment[-1][0] - segment[0][0])
    sampled_points = _resample_segment(dominant, fps, sample_fps)
    windows = _sequence_windows(sampled_points, sequence_length, max(1, sequence_length // 2))
    features = [extract_motion_features(window) for window in windows]
    summary = {
        "frames": frame_index,
        "fps": round(fps, 4),
        "dominant_track_frames": len(dominant),
        "sampled_points": len(sampled_points),
        "windows": len(features),
        "reason": None if features else "dominant track is too short",
    }
    return features, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Detection model not found: {args.model}")
    if args.sequence_length < 3 or args.sample_fps <= 0:
        raise ValueError("sequence length must be at least 3 and sample FPS must be positive")

    videos = [
        (label, class_id, path)
        for label, class_id in (("Normal", 0), ("Lame", 1))
        for path in sorted((args.data / label).glob("*.mp4"))
    ]
    if not videos:
        raise FileNotFoundError(f"No labeled MP4 videos found under: {args.data}")

    model = YOLO(str(args.model))
    all_features = []
    all_labels = []
    all_groups = []
    records = []
    for index, (label, class_id, video_path) in enumerate(videos, start=1):
        group = f"{label}/{video_path.name}"
        print(f"[{index:02d}/{len(videos)}] {group}", flush=True)
        try:
            features, summary = extract_video_features(
                video_path,
                model,
                args.sequence_length,
                args.sample_fps,
                args.confidence,
                args.imgsz,
            )
        except Exception as error:
            features = []
            summary = {"reason": str(error)}
        all_features.extend(features)
        all_labels.extend([class_id] * len(features))
        all_groups.extend([group] * len(features))
        records.append({"video": group, "label": class_id, **summary})
        print(f"  windows={len(features)} {summary.get('reason') or ''}", flush=True)

    if not all_features:
        raise RuntimeError("No usable cattle motion sequences were extracted.")

    args.output.mkdir(parents=True, exist_ok=True)
    feature_array = np.asarray(all_features, dtype=np.float64)
    label_array = np.asarray(all_labels, dtype=np.int64)
    group_array = np.asarray(all_groups)
    np.save(args.output / "features_X.npy", feature_array)
    np.save(args.output / "features_y.npy", label_array)
    np.save(args.output / "feature_groups.npy", group_array)

    metadata = {
        "feature_version": MOTION_FEATURE_VERSION,
        "sequence_length": args.sequence_length,
        "sample_fps": args.sample_fps,
        "confidence": args.confidence,
        "image_size": args.imgsz,
        "videos_found": len(videos),
        "videos_used": len(set(all_groups)),
        "sequences": len(feature_array),
        "class_sequences": {
            "Normal": int(np.sum(label_array == 0)),
            "Lame": int(np.sum(label_array == 1)),
        },
        "records": records,
    }
    (args.output / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: metadata[key] for key in metadata if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
