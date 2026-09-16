import numpy as np

from core.lameness_classifier import extract_motion_features


def test_motion_features_are_finite_and_have_stable_shape():
    points = np.array([[index * 0.01, 0.4] for index in range(16)])

    features = extract_motion_features(points)

    assert features.shape == (19,)
    assert np.isfinite(features).all()


def test_motion_features_do_not_depend_on_absolute_frame_position():
    points = np.array([
        [index * 0.01, 0.4 + np.sin(index) * 0.005]
        for index in range(16)
    ])

    original = extract_motion_features(points)
    translated = extract_motion_features(points + np.array([0.25, -0.15]))

    np.testing.assert_allclose(original, translated, atol=1e-12)


def test_motion_features_reject_malformed_sequences():
    with np.testing.assert_raises_regex(ValueError, "three 2D points"):
        extract_motion_features([[0.1, 0.2], [0.2, 0.3]])
