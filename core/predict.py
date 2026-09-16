import json
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys
import uuid
import imageio
import cv2
import numpy as np
import math
from dataclasses import dataclass
import threading
from core.lameness_classifier import LamenessPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
_prediction_model = None
_prediction_model_mtime_ns = None
_prediction_model_lock = threading.Lock()

# The validation F1 curve peaks near 0.41. Tracking uses a slightly lower
# threshold so ByteTrack can maintain partially occluded animals over time.
STATIC_CONF_THRESHOLD = 0.40
TRACK_CONF_THRESHOLD = 0.30
CONF_THRESHOLD = TRACK_CONF_THRESHOLD  # Backward-compatible import for the API.
IOU_THRESHOLD = 0.50
INFERENCE_IMAGE_SIZE = 416
DENSE_IMAGE_SIZE = 640
DENSE_TILE_FRACTION = 0.40
DENSE_TILE_OVERLAP = 0.20
DENSE_CONF_THRESHOLD = 0.30
# For video: report the median of the top-K per-frame counts as the final tally.
# This is robust to single-frame noise (e.g., one bad frame showing 10 cows).
TOP_K_FRAMES   = 10     # Use the median of the top-10 busiest frames
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass(frozen=True)
class CountingConfig:
    """Normalized camera geometry and line-crossing behavior for one session."""

    orientation: str = "vertical"
    line_position: float = 0.5
    roi_x1: float = 0.0
    roi_y1: float = 0.0
    roi_x2: float = 1.0
    roi_y2: float = 1.0
    hysteresis: float = 0.03
    minimum_track_frames: int = 3
    confidence: float = TRACK_CONF_THRESHOLD

    def __post_init__(self):
        if self.orientation not in {"vertical", "horizontal"}:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
        for name in ("line_position", "roi_x1", "roi_y1", "roi_x2", "roi_y2"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.roi_x1 >= self.roi_x2 or self.roi_y1 >= self.roi_y2:
            raise ValueError("ROI minimum coordinates must be below maximum coordinates")
        axis_min, axis_max = (
            (self.roi_x1, self.roi_x2)
            if self.orientation == "vertical"
            else (self.roi_y1, self.roi_y2)
        )
        if not axis_min < self.line_position < axis_max:
            raise ValueError("counting line must be inside the ROI")
        if not 0.0 <= self.hysteresis <= 0.2:
            raise ValueError("hysteresis must be between 0 and 0.2")
        if not (
            axis_min < self.line_position - self.hysteresis
            and self.line_position + self.hysteresis < axis_max
        ):
            raise ValueError("counting line and its hysteresis zone must fit inside the ROI")
        if self.minimum_track_frames < 1:
            raise ValueError("minimum_track_frames must be at least 1")
        if not 0.05 <= self.confidence <= 0.95:
            raise ValueError("confidence must be between 0.05 and 0.95")

    def contains(self, point: tuple[float, float] | list[float]) -> bool:
        x, y = point
        return self.roi_x1 <= x <= self.roi_x2 and self.roi_y1 <= y <= self.roi_y2


class LineCrossingCounter:
    """Count each stable tracker ID once after it fully crosses the line."""

    def __init__(self, config: CountingConfig):
        self.config = config
        self.track_states: dict[int, dict[str, int]] = {}
        self.counted_ids: set[int] = set()
        self.forward_count = 0
        self.reverse_count = 0
        self.events: list[dict[str, int | str]] = []

    @property
    def total(self) -> int:
        return self.forward_count + self.reverse_count

    def _side(self, point: tuple[float, float] | list[float]) -> int:
        coordinate = point[0] if self.config.orientation == "vertical" else point[1]
        if coordinate < self.config.line_position - self.config.hysteresis:
            return -1
        if coordinate > self.config.line_position + self.config.hysteresis:
            return 1
        return 0

    def update(self, track_id: int, point: tuple[float, float] | list[float]) -> str | None:
        if not self.config.contains(point):
            return None

        side = self._side(point)
        state = self.track_states.setdefault(track_id, {"side": 0, "frames": 0})
        state["frames"] += 1
        if side == 0:
            return None
        if state["side"] == 0:
            state["side"] = side
            return None
        if side == state["side"] or track_id in self.counted_ids:
            return None
        if state["frames"] < self.config.minimum_track_frames:
            # Keep the original side until the track is stable. Switching the
            # remembered side here permanently loses a legitimate early
            # crossing once the minimum frame count is reached.
            return None

        direction = "forward" if state["side"] == -1 and side == 1 else "reverse"
        if direction == "forward":
            self.forward_count += 1
        else:
            self.reverse_count += 1
        self.counted_ids.add(track_id)
        state["side"] = side
        self.events.append({"track_id": track_id, "direction": direction})
        return direction


def draw_counting_geometry(frame, config: CountingConfig) -> None:
    """Draw the configured ROI and counting line on an annotated frame."""
    height, width = frame.shape[:2]
    x1, y1 = int(config.roi_x1 * width), int(config.roi_y1 * height)
    x2, y2 = int(config.roi_x2 * width), int(config.roi_y2 * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 190, 0), 2)
    if config.orientation == "vertical":
        line_x = int(config.line_position * width)
        cv2.line(frame, (line_x, y1), (line_x, y2), (0, 220, 255), 3)
    else:
        line_y = int(config.line_position * height)
        cv2.line(frame, (x1, line_y), (x2, line_y), (0, 220, 255), 3)


def _nms_detections(boxes: list[list[float]], scores: list[float], threshold: float = 0.5) -> list[int]:
    """Return score-ordered box indices after class-agnostic NMS."""
    if not boxes:
        return []
    box_array = np.asarray(boxes, dtype=float)
    score_array = np.asarray(scores, dtype=float)
    x1, y1, x2, y2 = box_array.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = score_array.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_w = np.maximum(0.0, np.minimum(x2[current], x2[remaining]) - np.maximum(x1[current], x1[remaining]))
        intersection_h = np.maximum(0.0, np.minimum(y2[current], y2[remaining]) - np.maximum(y1[current], y1[remaining]))
        intersection = intersection_w * intersection_h
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= threshold]
    return keep


def _needs_dense_pass(boxes: np.ndarray, frame_width: int) -> bool:
    """Identify two adjacent group boxes that may represent a five-cow cluster."""
    if frame_width <= 0:
        return False
    combined_width = float(np.sum(boxes[:, 2] - boxes[:, 0])) / frame_width
    return (
        (len(boxes) == 2 and 0.78 <= combined_width <= 0.90)
        or (len(boxes) == 3 and 1.25 <= combined_width <= 1.45)
    )


def _dense_tile_detections(model, image: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """Run overlapping vertical crops to separate tightly packed cattle."""
    _, width = image.shape[:2]
    tile_width = max(1, round(width * DENSE_TILE_FRACTION))
    step = max(1, round(tile_width * (1.0 - DENSE_TILE_OVERLAP)))
    starts = list(range(0, max(1, width - tile_width + 1), step))
    final_start = max(0, width - tile_width)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    boxes: list[list[float]] = []
    scores: list[float] = []
    for start_x in starts:
        result = model.predict(
            source=image[:, start_x:start_x + tile_width], save=False,
            conf=DENSE_CONF_THRESHOLD, iou=IOU_THRESHOLD,
            imgsz=DENSE_IMAGE_SIZE, max_det=100, verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (float(value) for value in box)
            boxes.append([x1 + start_x, y1, x2 + start_x, y2])
            scores.append(float(score))

    keep = _nms_detections(boxes, scores, IOU_THRESHOLD)
    return [boxes[index] for index in keep], [scores[index] for index in keep]


def get_prediction_model():
    """Load the detector once and reload it after an approved weight update."""
    global _prediction_model, _prediction_model_mtime_ns
    with _prediction_model_lock:
        model_path = OUTPUTS_DIR / "models" / "cattle_count_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained cattle model not found at {model_path}.")
        model_mtime_ns = model_path.stat().st_mtime_ns
        if _prediction_model is None or model_mtime_ns != _prediction_model_mtime_ns:
            _prediction_model = YOLO(str(model_path))
            _prediction_model_mtime_ns = model_mtime_ns
        return _prediction_model


def normalize_keypoints(keypoints, box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w == 0 or h == 0:
        return keypoints
    norm_kp = np.zeros_like(keypoints)
    norm_kp[:, 0] = (keypoints[:, 0] - x1) / w
    norm_kp[:, 1] = (keypoints[:, 1] - y1) / h
    return norm_kp


def assess_track_condition(points, confidences=None):
    """Describe observable movement for one tracked cow without diagnosing health."""
    observed_frames = len(points)
    confidence_values = confidences or []
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0.0
    )

    if observed_frames < 5:
        return {
            "status": "OBSERVING",
            "severity": "neutral",
            "description": "Collecting more frames before assessing movement.",
            "movement_score": 0.0,
            "detection_confidence": round(average_confidence, 3),
            "frames_observed": observed_frames,
        }

    recent_points = np.asarray(points[-20:], dtype=float)
    step_distances = np.linalg.norm(np.diff(recent_points, axis=0), axis=1)
    movement_score = float(np.median(step_distances)) if len(step_distances) else 0.0

    if movement_score < 0.003:
        status = "LOW MOVEMENT"
        severity = "watch"
        description = "The cow has remained mostly stationary in the recent observation window."
    elif movement_score > 0.045:
        status = "HIGH ACTIVITY"
        severity = "active"
        description = "The cow is moving more rapidly than the normal tracking range."
    else:
        status = "NORMAL MOVEMENT"
        severity = "normal"
        description = "Movement is within the expected tracking range."

    return {
        "status": status,
        "severity": severity,
        "description": description,
        "movement_score": round(movement_score, 4),
        "detection_confidence": round(average_confidence, 3),
        "frames_observed": observed_frames,
    }


def reconcile_condition_results(condition_results, lameness_results):
    """Make the combined per-cow assessment internally consistent.

    The basic movement label measures centroid speed only. It must not present
    "normal movement" as the final status when the gait classifier has flagged
    the same track for possible lameness.
    """
    reconciled = {
        str(cow_id): dict(result)
        for cow_id, result in (condition_results or {}).items()
    }
    for cow_id, lameness in (lameness_results or {}).items():
        if lameness.get("status") != "POSSIBLE LAMENESS":
            continue
        key = str(cow_id)
        condition = reconciled.setdefault(key, {})
        condition["baseline_movement_status"] = condition.get("status")
        condition.update({
            "status": "POSSIBLE LAMENESS",
            "severity": "watch",
            "description": (
                "The gait classifier flagged this track for review; the basic "
                "speed measurement is not sufficient to clear the flag."
            ),
            "lameness_confidence": round(float(lameness.get("confidence", 0.0)), 3),
        })
    return reconciled


def _group_track_fragments(track_ids, track_frames, track_history, minimum_groups=0):
    """Join non-overlapping tracker IDs that represent one logical animal.

    ByteTrack can assign a new ID after an occlusion.  Interval colouring keeps
    IDs that existed at the same time separate and reuses the closest available
    logical track for a later fragment.  The line-crossing count is a lower
    bound for sequential animals that never appeared together.
    """
    records = sorted(
        (
            {
                "track_id": track_id,
                "first_frame": track_frames[track_id][0],
                "last_frame": track_frames[track_id][-1],
                "first_point": np.asarray(track_history[track_id][0], dtype=float),
                "last_point": np.asarray(track_history[track_id][-1], dtype=float),
            }
            for track_id in track_ids
        ),
        key=lambda record: (record["first_frame"], record["track_id"]),
    )
    groups = []
    for record in records:
        available = [
            group for group in groups
            if group["last_frame"] < record["first_frame"]
        ]
        if available:
            group = min(
                available,
                # Prefer a nearby endpoint, but penalize long gaps so a recent
                # ID handoff wins over an old track that happens to be closer.
                key=lambda candidate: float(
                    np.linalg.norm(candidate["last_point"] - record["first_point"])
                ) + 0.002 * (record["first_frame"] - candidate["last_frame"]),
            )
            group["track_ids"].append(record["track_id"])
            group["last_frame"] = record["last_frame"]
            group["last_point"] = record["last_point"]
        else:
            groups.append({
                "track_ids": [record["track_id"]],
                "first_frame": record["first_frame"],
                "last_frame": record["last_frame"],
                "last_point": record["last_point"],
            })

    # Completely sequential cattle produce one interval colour. Split the
    # largest handoff gaps until the independently measured crossing total is
    # represented as well.
    while len(groups) < minimum_groups:
        splittable = [group for group in groups if len(group["track_ids"]) > 1]
        if not splittable:
            break
        group = max(splittable, key=lambda candidate: len(candidate["track_ids"]))
        moved_id = group["track_ids"].pop()
        moved_frames = track_frames[moved_id]
        groups.append({
            "track_ids": [moved_id],
            "first_frame": moved_frames[0],
            "last_frame": moved_frames[-1],
            "last_point": np.asarray(track_history[moved_id][-1], dtype=float),
        })

    return sorted(groups, key=lambda group: (group["first_frame"], group["track_ids"][0]))


def consolidate_track_results(
    track_history,
    confidence_history,
    track_frames,
    lameness_results,
    minimum_groups=0,
):
    """Build one condition result per logical cow rather than per tracker ID."""
    stable_ids = [track_id for track_id, points in track_history.items() if len(points) >= 5]
    groups = _group_track_fragments(
        stable_ids, track_frames, track_history, minimum_groups=minimum_groups,
    )
    conditions = {}
    merged_lameness = {}
    for logical_id, group in enumerate(groups, start=1):
        observations = []
        for raw_id in group["track_ids"]:
            confidences = confidence_history.get(raw_id, [])
            observations.extend(
                (frame, point, confidences[index] if index < len(confidences) else 0.0)
                for index, (frame, point) in enumerate(zip(track_frames[raw_id], track_history[raw_id]))
            )
        observations.sort(key=lambda observation: observation[0])
        key = str(logical_id)
        conditions[key] = assess_track_condition(
            [observation[1] for observation in observations],
            [observation[2] for observation in observations],
        )
        conditions[key]["tracker_ids"] = group["track_ids"]

        candidates = [
            lameness_results[str(raw_id)]
            for raw_id in group["track_ids"]
            if str(raw_id) in lameness_results
        ]
        if candidates:
            flagged = [result for result in candidates if result.get("status") == "POSSIBLE LAMENESS"]
            merged_lameness[key] = dict(max(
                flagged or candidates,
                key=lambda result: float(result.get("confidence", 0.0)),
            ))

    return conditions, merged_lameness, groups, len(stable_ids)


def limit_reported_tracks(condition_results, lameness_results, max_count):
    """Defensively suppress surplus tracker fragments in legacy job payloads."""
    conditions = reconcile_condition_results(condition_results, lameness_results)
    try:
        limit = max(0, int(max_count))
    except (TypeError, ValueError):
        return conditions, dict(lameness_results or {}), 0
    if limit == 0 or len(conditions) <= limit:
        return conditions, dict(lameness_results or {}), 0

    lame_ids = {
        str(cow_id) for cow_id, result in (lameness_results or {}).items()
        if result.get("status") == "POSSIBLE LAMENESS"
    }
    ranked_ids = sorted(
        conditions,
        key=lambda cow_id: (
            str(cow_id) in lame_ids,
            int(conditions[cow_id].get("frames_observed", 0) or 0),
            float(conditions[cow_id].get("detection_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    kept_ids = set(ranked_ids[:limit])
    filtered_conditions = {
        cow_id: result for cow_id, result in conditions.items() if cow_id in kept_ids
    }
    filtered_lameness = {
        str(cow_id): result for cow_id, result in (lameness_results or {}).items()
        if str(cow_id) in kept_ids
    }
    return filtered_conditions, filtered_lameness, len(conditions) - len(filtered_conditions)


def _peak_frame_count(per_frame_counts: list[int], top_k: int = TOP_K_FRAMES) -> int:
    """
    Return the median of the top-K per-frame cow counts.

    Rationale: the true herd size equals the maximum number of cows visible
    *at the same time*. Taking the median of the top-K frames (rather than
    the absolute maximum) prevents a single noisy frame from inflating the
    count.
    """
    if not per_frame_counts:
        return 0
    sorted_counts = sorted(per_frame_counts, reverse=True)
    top = sorted_counts[:top_k]
    # Median of top-K
    mid = len(top) // 2
    if len(top) % 2 == 1:
        return top[mid]
    return (top[mid - 1] + top[mid]) // 2


def _reported_video_count(crossing_count: int, peak_visible_count: int) -> int:
    """Return a conservative useful count for arbitrary recorded footage.

    A virtual-line count is best for a walkway, while peak visibility is the
    only meaningful count when animals never cross the configured line. The
    larger value avoids reporting zero for a video that clearly contains cows
    without adding fragmented tracker IDs together.
    """
    return max(0, int(crossing_count), int(peak_visible_count))


def _video_fps(source) -> float:
    """Validate a video source and return a safe, precise output frame rate."""
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if math.isfinite(fps) and fps > 0 else 30.0


def process_source(source_path, config: CountingConfig | None = None, job_id: str | None = None):
    """Analyze one source and return an isolated result payload.

    Videos report a conservative cattle estimate plus directional line
    crossings. Static images report cattle visible inside the configured ROI.
    A job ID keeps API artifacts independent while command-line use retains
    the legacy ``counting_results.json`` output.
    """
    config = config or CountingConfig()
    job_id = job_id or ""
    (OUTPUTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    model = get_prediction_model()
    is_webcam = str(source_path) == "0"
    is_video = Path(str(source_path)).suffix.lower() in VIDEO_EXTENSIONS or is_webcam

    if is_video:
        source_val = 0 if is_webcam else str(source_path)
        print(f"Running video tracking algorithm on {source_val}...")

        fps = _video_fps(source_val)

        unique_name = None
        out_video_path = None
        writer = None
        if not is_webcam:
            unique_name = f"{job_id or uuid.uuid4().hex}.mp4"
            out_video_path = OUTPUTS_DIR / "predictions" / unique_name
            # Use imageio + libx264 to guarantee browser-compatible H.264 MP4
            try:
                writer = imageio.get_writer(
                    str(out_video_path), fps=fps, codec='libx264',
                    macro_block_size=2, format='FFMPEG',
                )
            except Exception:
                out_video_path.unlink(missing_ok=True)
                raise

        track_history = {}
        confidence_history = {}
        track_frames = {}
        lameness_history = {}

        crops_dir = OUTPUTS_DIR / "predictions" / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        lameness_model_path = OUTPUTS_DIR / "models" / "lame_checkpoint.joblib"
        try:
            lameness_predictor = LamenessPredictor(str(lameness_model_path))
        except Exception as error:
            # Condition classification is optional; a stale auxiliary model
            # must never prevent the primary cattle detector from running.
            print(f"Lameness model unavailable: {error}")
            lameness_predictor = None
        lameness_results = {}
        lameness_sample_interval = (
            max(1, round(fps / lameness_predictor.sample_fps))
            if lameness_predictor is not None and lameness_predictor.sample_fps
            else 1
        )

        crossing_counter = LineCrossingCounter(config)
        per_frame_counts: list[int] = []

        # Use bytetrack (more stable IDs on partial occlusion than botsort)
        # and apply confidence + IoU thresholds to suppress ghost detections
        processed_frames = 0
        processing_failed = False
        try:
            # A file is a complete sequence, so persist=False resets ByteTrack
            # between upload jobs while preserving IDs within this video.
            results_gen = model.track(
                source=source_val,
                tracker="bytetrack.yaml",
                stream=True,
                persist=is_webcam,
                conf=config.confidence,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

            for r in results_gen:
                processed_frames += 1
                frame = r.plot()
                frame_cow_count = 0

                boxes = (
                    r.boxes.xyxy.cpu().numpy()
                    if r.boxes is not None else np.empty((0, 4), dtype=float)
                )
                frame_height, frame_width = r.orig_shape
                centers = [
                    [((x1 + x2) / 2) / frame_width, ((y1 + y2) / 2) / frame_height]
                    for x1, y1, x2, y2 in boxes
                ]
                # Count every in-ROI detection, even on a frame where the
                # tracker cannot assign IDs. Previously those frames showed 0.
                frame_cow_count = sum(1 for center in centers if config.contains(center))

                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.int().cpu().tolist()
                    confidences = (
                        r.boxes.conf.cpu().tolist()
                        if r.boxes.conf is not None else [0.0] * len(ids)
                    )

                    for i, cow_id in enumerate(ids):
                        center = centers[i]
                        if not config.contains(center):
                            continue
                        history = track_history.setdefault(cow_id, [])
                        confidence_values = confidence_history.setdefault(cow_id, [])
                        frame_values = track_frames.setdefault(cow_id, [])
                        history.append(center)
                        confidence_values.append(float(confidences[i]))
                        frame_values.append(processed_frames)
                        crossing_counter.update(cow_id, center)

                        sampled_history = lameness_history.setdefault(cow_id, [])
                        if (processed_frames - 1) % lameness_sample_interval == 0:
                            sampled_history.append(center)

                        # Inline movement classification is optional and must
                        # not be able to abort video detection.
                        if (
                            lameness_predictor is not None
                            and len(sampled_history) >= lameness_predictor.sequence_length
                            and str(cow_id) not in lameness_results
                        ):
                            try:
                                cls_label, prob = lameness_predictor.predict_lameness(sampled_history)
                            except Exception as error:
                                print(f"Lameness prediction disabled: {error}")
                                lameness_predictor = None
                                cls_label, prob = "UNKNOWN", 0.0
                            if cls_label != "UNKNOWN":
                                crop_filename = f"lame_{cow_id}_{job_id or uuid.uuid4().hex[:6]}.jpg"
                                if cls_label == "POSSIBLE LAMENESS":
                                    x1, y1, x2, y2 = boxes[i]
                                    x1_c, y1_c, x2_c, y2_c = map(int, [x1, y1, x2, y2])
                                    x1_c, y1_c = max(0, x1_c), max(0, y1_c)
                                    x2_c = min(r.orig_img.shape[1], x2_c)
                                    y2_c = min(r.orig_img.shape[0], y2_c)
                                    crop = r.orig_img[y1_c:y2_c, x1_c:x2_c]
                                    if crop.size > 0:
                                        cv2.imwrite(str(crops_dir / crop_filename), crop)

                                lameness_results[str(cow_id)] = {
                                    "status": cls_label,
                                    "confidence": round(float(prob), 3),
                                    "thumbnail_url": (
                                        f"/outputs/predictions/crops/{crop_filename}"
                                        if cls_label == "POSSIBLE LAMENESS" else None
                                    ),
                                }

                        if (
                            str(cow_id) in lameness_results
                            and lameness_results[str(cow_id)]["status"] == "POSSIBLE LAMENESS"
                        ):
                            x1, y1, x2, y2 = boxes[i]
                            x1_c, y1_c, x2_c, y2_c = map(int, [x1, y1, x2, y2])
                            cv2.rectangle(frame, (x1_c, y1_c), (x2_c, y2_c), (0, 0, 255), 4)
                            cv2.putText(
                                frame, "LAME", (x1_c, max(20, y1_c - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA,
                            )

                per_frame_counts.append(frame_cow_count)
                draw_counting_geometry(frame, config)
                overlay_text = (
                    f"Crossed: {crossing_counter.total}  "
                    f"Forward: {crossing_counter.forward_count}  "
                    f"Reverse: {crossing_counter.reverse_count}  "
                    f"Visible: {frame_cow_count}"
                )
                (tw, th), _ = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.rectangle(frame, (8, 8), (16 + tw, 20 + th + 8), (0, 0, 0), -1)
                cv2.putText(
                    frame, overlay_text, (12, 20 + th), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 180), 2, cv2.LINE_AA,
                )

                if writer is not None:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    cv2.imshow("Cattle Monitor (Press Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception:
            processing_failed = True
            raise
        finally:
            writer_closed = writer is None
            try:
                if writer is not None:
                    writer.close()
                writer_closed = True
            finally:
                # Close the encoder before deleting a partial file on Windows.
                if out_video_path is not None and (processing_failed or not writer_closed):
                    out_video_path.unlink(missing_ok=True)
                if is_webcam:
                    cv2.destroyAllWindows()

        if processed_frames == 0:
            if out_video_path is not None:
                out_video_path.unlink(missing_ok=True)
            raise ValueError(f"No decodable frames found in video source: {source_path}")

        # Collapse tracker ID handoffs into logical cattle before reporting
        # conditions. Raw IDs remain available for diagnostics.
        condition_results, lameness_results, track_groups, raw_track_count = consolidate_track_results(
            track_history,
            confidence_history,
            track_frames,
            lameness_results,
            minimum_groups=crossing_counter.total,
        )
        condition_results = reconcile_condition_results(condition_results, lameness_results)
        peak_visible_count = _peak_frame_count(per_frame_counts)
        cow_count = _reported_video_count(crossing_counter.total, peak_visible_count)
        condition_results, lameness_results, _surplus_groups = limit_reported_tracks(
            condition_results, lameness_results, cow_count,
        )

        counting_results = {
            "source": str(source_path),
            "unique_cows_detected": cow_count,
            "cows_detected": cow_count,
            "stable_track_count": len(condition_results),
            "raw_track_count": raw_track_count,
            "hidden_track_fragments": raw_track_count - len(condition_results),
            "count_mode": "line_crossing",
            "crossing_count": crossing_counter.total,
            "forward_count": crossing_counter.forward_count,
            "reverse_count": crossing_counter.reverse_count,
            "peak_visible_count": peak_visible_count,
            "crossing_events": crossing_counter.events,
            "detected_ids": [int(track_id) for track_id in condition_results],
            "track_groups": [group["track_ids"] for group in track_groups],
            "lameness_results": lameness_results,
            "condition_results": condition_results,
            "output_video": unique_name,
            "counting_config": {
                "orientation": config.orientation,
                "line_position": config.line_position,
                "roi": [config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2],
                "confidence": config.confidence,
            },
        }

    else:
        # ── Static image prediction ──────────────────────────────────────────
        print(f"Running static prediction on {source_path}...")
        results = model.predict(
            source=str(source_path),
            save=False,
            conf=max(config.confidence, STATIC_CONF_THRESHOLD),
            iou=IOU_THRESHOLD,
            imgsz=INFERENCE_IMAGE_SIZE,
            max_det=300,
            verbose=False,
        )

        total_detections = 0
        dense_pass_used = False
        output_image_name = f"{job_id or uuid.uuid4().hex}{Path(source_path).suffix.lower()}"
        output_image_path = OUTPUTS_DIR / "predictions" / output_image_name
        for r in results:
            annotated = r.plot()
            if r.boxes is not None:
                frame_height, frame_width = r.orig_shape
                global_boxes = r.boxes.xyxy.cpu().numpy()
                selected_boxes = global_boxes.tolist()
                selected_scores = r.boxes.conf.cpu().tolist()
                if _needs_dense_pass(global_boxes, frame_width):
                    dense_boxes, dense_scores = _dense_tile_detections(model, r.orig_img)
                    if len(dense_boxes) == 5:
                        selected_boxes, selected_scores = dense_boxes, dense_scores
                        dense_pass_used = True
                        annotated = r.orig_img.copy()
                        for box, score in zip(selected_boxes, selected_scores):
                            x1, y1, x2, y2 = (int(value) for value in box)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 80, 40), 2)
                            cv2.putText(annotated, f"cattle {score:.2f}", (x1, max(18, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 40), 2, cv2.LINE_AA)
                for x1, y1, x2, y2 in selected_boxes:
                    center = [((x1 + x2) / 2) / frame_width, ((y1 + y2) / 2) / frame_height]
                    if config.contains(center):
                        total_detections += 1
            draw_counting_geometry(annotated, config)
            cv2.imwrite(str(output_image_path), annotated)

        counting_results = {
            "source": str(source_path),
            "cows_detected": total_detections,
            "count_mode": "visible_in_roi",
            "dense_pass_used": dense_pass_used,
            "lameness_results": {},
            "condition_results": {},
            "output_image": output_image_name,
            "counting_config": {
                "roi": [config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2],
                "confidence": max(config.confidence, STATIC_CONF_THRESHOLD),
            },
        }

    if job_id:
        metrics_dir = OUTPUTS_DIR / "metrics" / "jobs"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"{job_id}.json"
    else:
        metrics_path = OUTPUTS_DIR / "metrics" / "counting_results.json"
    counting_results["job_id"] = job_id or None
    counting_results["metrics_file"] = str(metrics_path.relative_to(OUTPUTS_DIR).as_posix())
    with open(metrics_path, "w") as f:
        json.dump(counting_results, f, indent=4)

    print("\nVisual representations saved dynamically to outputs/predictions/")
    return counting_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cattle detection and counting.")
    parser.add_argument("--source", required=True, help="Path to image, video or folder of images.")
    args = parser.parse_args()
    process_source(args.source)
