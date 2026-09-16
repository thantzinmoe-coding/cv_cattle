from backend.routers.predict import _reported_cow_count


def test_historical_video_jobs_fall_back_to_peak_visible_count():
    result = {
        "unique_cows_detected": 0,
        "crossing_count": 0,
        "peak_visible_count": 3,
    }

    assert _reported_cow_count(result) == 3


def test_explicit_new_video_count_takes_precedence():
    result = {
        "cows_detected": 4,
        "crossing_count": 2,
        "peak_visible_count": 3,
    }

    assert _reported_cow_count(result) == 4


def test_image_count_is_unchanged():
    assert _reported_cow_count({"cows_detected": 2}) == 2
