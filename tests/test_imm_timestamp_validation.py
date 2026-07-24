import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker


_INVALID_TIMESTAMPS = [np.nan, np.inf, -np.inf, "not-a-time", True, np.array([0.0])]


@pytest.mark.parametrize("time_s", _INVALID_TIMESTAMPS)
def test_imm_tracker_rejects_invalid_initial_timestamp(time_s):
    with pytest.raises(ValueError, match="initial_time_s must be a finite numeric timestamp"):
        AsyncInteractingMultipleModelTracker(np.zeros(3), time_s)


@pytest.mark.parametrize("time_s", _INVALID_TIMESTAMPS)
def test_imm_tracker_rejects_invalid_prediction_timestamp(time_s):
    tracker = AsyncInteractingMultipleModelTracker(np.zeros(3), 0.0)

    with pytest.raises(ValueError, match="time_s must be a finite numeric timestamp"):
        tracker.predict_to(time_s)


def test_imm_tracker_accepts_numeric_string_timestamps():
    tracker = AsyncInteractingMultipleModelTracker(np.zeros(3), "0.0")

    tracker.predict_to("1.25")

    assert tracker.current_time_s == 1.25
