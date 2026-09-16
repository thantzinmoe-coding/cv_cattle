"""Train and evaluate the deployed lameness classifier on video groups."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "outputs" / "dataset_features"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "lame_checkpoint.joblib"


def _video_level_predictions(model, features, labels, groups):
    probabilities = model.predict_proba(features)
    lame_index = int(np.flatnonzero(model.classes_ == 1)[0])
    video_labels = []
    video_predictions = []
    video_probabilities = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        probability = float(np.mean(probabilities[mask, lame_index]))
        video_probabilities[group] = round(probability, 6)
        video_labels.append(int(labels[mask][0]))
        video_predictions.append(int(probability >= 0.5))
    return np.asarray(video_labels), np.asarray(video_predictions), video_probabilities


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features = np.load(args.features / "features_X.npy")
    labels = np.load(args.features / "features_y.npy")
    groups = np.load(args.features / "feature_groups.npy")
    metadata = json.loads((args.features / "feature_metadata.json").read_text(encoding="utf-8"))
    if not (len(features) == len(labels) == len(groups)) or len(features) < 4:
        raise ValueError("Feature, label, and video-group arrays are inconsistent.")

    unique_groups = np.asarray(sorted(set(groups.tolist())))
    group_labels = np.asarray([int(labels[groups == group][0]) for group in unique_groups])
    if len(set(group_labels.tolist())) != 2:
        raise ValueError("Both Normal and Lame videos are required.")
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=group_labels,
    )
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    parameters = {
        "n_estimators": 500,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": args.seed,
        "n_jobs": -1,
    }
    evaluation_model = RandomForestClassifier(**parameters)
    evaluation_model.fit(features[train_mask], labels[train_mask])

    sequence_predictions = evaluation_model.predict(features[test_mask])
    video_truth, video_predictions, video_probabilities = _video_level_predictions(
        evaluation_model,
        features[test_mask],
        labels[test_mask],
        groups[test_mask],
    )
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "feature_version": metadata["feature_version"],
        "sequence_length": metadata["sequence_length"],
        "sample_fps": metadata["sample_fps"],
        "train_videos": train_groups.tolist(),
        "test_videos": test_groups.tolist(),
        "train_sequences": int(np.sum(train_mask)),
        "test_sequences": int(np.sum(test_mask)),
        "sequence_accuracy": float(accuracy_score(labels[test_mask], sequence_predictions)),
        "video_accuracy": float(accuracy_score(video_truth, video_predictions)),
        "video_confusion_matrix": confusion_matrix(
            video_truth, video_predictions, labels=[0, 1]
        ).tolist(),
        "video_classification_report": classification_report(
            video_truth,
            video_predictions,
            labels=[0, 1],
            target_names=["Normal", "Lame"],
            zero_division=0,
            output_dict=True,
        ),
        "test_video_lame_probabilities": video_probabilities,
    }

    # Refit on every labeled video only after producing the held-out report.
    deployed_model = RandomForestClassifier(**parameters)
    deployed_model.fit(features, labels)
    artifact = {
        "model": deployed_model,
        "feature_version": metadata["feature_version"],
        "sequence_length": metadata["sequence_length"],
        "sample_fps": metadata["sample_fps"],
        "threshold": 0.5,
        "classes": {0: "Normal", 1: "Lame"},
        "trained_at": report["trained_at"],
        "sklearn_version": sklearn.__version__,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        backup_path = args.output.with_name(f"{args.output.stem}.previous{args.output.suffix}")
        shutil.copy2(args.output, backup_path)
        report["previous_model_backup"] = str(backup_path)
    temporary_path = args.output.with_suffix(args.output.suffix + ".part")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Video-level accuracy: {report['video_accuracy']:.3f}")
    print(f"Sequence-level accuracy: {report['sequence_accuracy']:.3f}")
    print(f"Confusion matrix [Normal, Lame]: {report['video_confusion_matrix']}")
    print(f"Saved model: {args.output}")
    print(f"Saved evaluation: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
