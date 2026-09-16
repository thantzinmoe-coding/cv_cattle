import numpy as np
import joblib
from pathlib import Path


MOTION_FEATURE_VERSION = 2
DEFAULT_SEQUENCE_LENGTH = 16
DEFAULT_SAMPLE_FPS = 10.0


def extract_motion_features(points):
    """Build translation-invariant gait features from normalized centers."""
    sequence = np.asarray(points, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 2 or len(sequence) < 3:
        raise ValueError("A motion sequence must contain at least three 2D points.")
    if not np.isfinite(sequence).all():
        raise ValueError("Motion sequence contains non-finite coordinates.")

    steps = np.diff(sequence, axis=0)
    speeds = np.linalg.norm(steps, axis=1)
    displacement = sequence[-1] - sequence[0]
    displacement_norm = float(np.linalg.norm(displacement))
    path_length = float(np.sum(speeds))
    straightness = displacement_norm / max(path_length, 1e-8)

    if displacement_norm > 1e-6:
        forward_axis = displacement / displacement_norm
    else:
        centered = sequence - sequence.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        forward_axis = axes[0]
    lateral_axis = np.array([-forward_axis[1], forward_axis[0]])

    centered = sequence - sequence[0]
    forward_motion = centered @ forward_axis
    lateral_motion = centered @ lateral_axis
    time_axis = np.linspace(0.0, 1.0, len(sequence))

    def detrended(values):
        slope, intercept = np.polyfit(time_axis, values, 1)
        return values - ((slope * time_axis) + intercept)

    forward_residual = detrended(forward_motion)
    lateral_residual = detrended(lateral_motion)
    acceleration = np.diff(steps, axis=0)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)

    if len(steps) > 1:
        dot_products = np.sum(steps[:-1] * steps[1:], axis=1)
        denominators = np.linalg.norm(steps[:-1], axis=1) * np.linalg.norm(steps[1:], axis=1)
        cosines = np.divide(
            dot_products,
            denominators,
            out=np.ones_like(dot_products),
            where=denominators > 1e-8,
        )
        direction_change = float(np.mean(np.clip(cosines, -1.0, 1.0) < 0.5))
    else:
        direction_change = 0.0

    moving_speed = max(float(np.median(speeds)), 1e-4)
    features = np.array([
        abs(float(displacement[0])),
        abs(float(displacement[1])),
        displacement_norm,
        path_length,
        straightness,
        float(np.mean(speeds)),
        float(np.median(speeds)),
        float(np.std(speeds)),
        float(np.quantile(speeds, 0.90)),
        float(np.max(speeds)),
        float(np.std(steps[:, 0])),
        float(np.std(steps[:, 1])),
        float(np.std(forward_residual)),
        float(np.std(lateral_residual)),
        float(np.ptp(lateral_residual)),
        float(np.mean(acceleration_norm)),
        float(np.std(acceleration_norm)),
        direction_change,
        float(np.mean(speeds < moving_speed * 0.25)),
    ], dtype=np.float64)
    return features


class LamenessPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.feature_version = 1
        self.sequence_length = 30
        self.sample_fps = None
        self.threshold = 0.5
        if Path(model_path).exists():
            artifact = joblib.load(model_path)
            if isinstance(artifact, dict) and "model" in artifact:
                self.model = artifact["model"]
                self.feature_version = int(artifact.get("feature_version", MOTION_FEATURE_VERSION))
                self.sequence_length = int(artifact.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
                self.sample_fps = float(artifact.get("sample_fps", DEFAULT_SAMPLE_FPS))
                self.threshold = float(artifact.get("threshold", 0.5))
            else:
                # Backward compatibility with the previous six-feature model.
                self.model = artifact
            print(f"Lameness model loaded from {model_path}")
        else:
            print(f"Lameness model not found at {model_path}. Predictions will be UNKNOWN.")

    def is_ready(self):
        return self.model is not None

    def predict_lameness(self, point_sequence):
        if not self.is_ready() or len(point_sequence) < self.sequence_length:
            return "UNKNOWN", 0.0

        sequence = np.asarray(point_sequence[-self.sequence_length:], dtype=np.float64)
        if self.feature_version == MOTION_FEATURE_VERSION:
            features = extract_motion_features(sequence)
        else:
            variance = np.var(sequence, axis=0).flatten()
            mean = np.mean(sequence, axis=0).flatten()
            displacement = (sequence[-1] - sequence[0]).flatten()
            features = np.concatenate([variance, mean, displacement])

        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        classes = np.asarray(self.model.classes_)
        lame_indices = np.flatnonzero(classes == 1)
        if len(lame_indices) != 1:
            return "UNKNOWN", 0.0
        lame_probability = float(probabilities[int(lame_indices[0])])

        if lame_probability >= self.threshold:
            return "POSSIBLE LAMENESS", lame_probability
        return "NORMAL", 1.0 - lame_probability
