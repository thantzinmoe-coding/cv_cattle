import pytest
import numpy as np

from core.predict import (
    CountingConfig,
    LineCrossingCounter,
    _needs_dense_pass,
    _nms_detections,
    assess_track_condition,
)


def test_condition_waits_for_enough_tracking_frames():
    result = assess_track_condition([[0.1, 0.1]] * 4, [0.8] * 4)
    assert result["status"] == "OBSERVING"
    assert result["frames_observed"] == 4


def test_stationary_track_is_flagged_for_observation():
    result = assess_track_condition([[0.2, 0.2]] * 8, [0.9] * 8)
    assert result["status"] == "LOW MOVEMENT"
    assert result["severity"] == "watch"
    assert result["detection_confidence"] == 0.9


def test_expected_motion_is_reported_as_normal():
    points = [[0.1 + (index * 0.01), 0.2] for index in range(8)]
    result = assess_track_condition(points, [0.85] * 8)
    assert result["status"] == "NORMAL MOVEMENT"


def test_large_motion_is_reported_as_high_activity():
    points = [[0.1 + (index * 0.08), 0.2] for index in range(8)]
    result = assess_track_condition(points, [0.85] * 8)
    assert result["status"] == "HIGH ACTIVITY"


def test_line_counter_counts_one_track_only_once_after_full_crossing():
    counter = LineCrossingCounter(CountingConfig(minimum_track_frames=3, hysteresis=0.03))

    assert counter.update(7, (0.30, 0.50)) is None
    assert counter.update(7, (0.47, 0.50)) is None  # Inside hysteresis zone.
    assert counter.update(7, (0.70, 0.50)) == "forward"
    assert counter.update(7, (0.30, 0.50)) is None  # Hesitation cannot double-count.
    assert counter.total == 1
    assert counter.forward_count == 1
    assert counter.reverse_count == 0


def test_line_counter_tracks_both_directions_across_different_ids():
    counter = LineCrossingCounter(CountingConfig(minimum_track_frames=2))

    counter.update(1, (0.25, 0.5))
    assert counter.update(1, (0.75, 0.5)) == "forward"
    counter.update(2, (0.75, 0.5))
    assert counter.update(2, (0.25, 0.5)) == "reverse"
    assert counter.total == 2


def test_line_counter_ignores_tracks_outside_roi():
    config = CountingConfig(roi_x1=0.2, roi_y1=0.2, roi_x2=0.8, roi_y2=0.8)
    counter = LineCrossingCounter(config)

    counter.update(3, (0.1, 0.5))
    counter.update(3, (0.9, 0.5))
    assert counter.total == 0


def test_counting_config_rejects_invalid_roi():
    with pytest.raises(ValueError, match="ROI"):
        CountingConfig(roi_x1=0.8, roi_x2=0.2)


def test_dense_pass_only_triggers_for_a_few_wide_boxes():
    crowded = np.array([[0, 0, 400, 300], [350, 0, 750, 300]], dtype=float)
    ordinary = np.array([[0, 0, 100, 300], [350, 0, 450, 300]], dtype=float)
    assert _needs_dense_pass(crowded, 1000)
    assert not _needs_dense_pass(ordinary, 1000)
    assert not _needs_dense_pass(crowded[:1], 1000)


def test_nms_removes_overlapping_duplicate_boxes():
    boxes = [[0, 0, 100, 100], [5, 5, 105, 105], [200, 0, 300, 100]]
    assert _nms_detections(boxes, [0.9, 0.8, 0.7], threshold=0.5) == [0, 2]
