import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import (
    nearest_candidate_to_truth,
    sweep_positions_against_truth,
    sweep_radar_against_truth,
)


def _origin_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


def test_position_sweep_keeps_large_representable_norm_finite():
    positions = np.array([[6.0e307, 8.0e307, 0.0]])

    with np.errstate(over="raise", invalid="raise"):
        sweep = sweep_positions_against_truth(
            measurement_times_s=np.array([0.0]),
            measurement_positions_m=positions,
            truth=_origin_truth(),
            taus_s=[0.0],
            dimensions=2,
            max_truth_time_delta_s=1.0,
        )

    error = float(sweep.loc[0, "mean_error_m"])
    assert np.isfinite(error)
    assert error == pytest.approx(1.0e308)


def test_radar_sweep_keeps_large_representable_norm_finite():
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [1],
            "track_id": [7],
            "east_m": [6.0e307],
            "north_m": [8.0e307],
            "up_m": [0.0],
            "cat_prob_uav": [0.9],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        sweep = sweep_radar_against_truth(
            radar=radar,
            truth=_origin_truth(),
            taus_s=[0.0],
            dimensions=2,
            selection="highest-catprob",
            catprob_threshold=0.4,
            max_truth_time_delta_s=1.0,
        )

    error = float(sweep.loc[0, "mean_error_m"])
    assert np.isfinite(error)
    assert error == pytest.approx(1.0e308)


def test_nearest_candidate_uses_large_representable_norms():
    candidates = pd.DataFrame(
        {
            "track_id": [1, 2],
            "east_m": [6.0e307, 9.0e307],
            "north_m": [8.0e307, 9.0e307],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        selected = nearest_candidate_to_truth(
            candidates,
            np.zeros(3, dtype=float),
        )

    assert selected is not None
    assert int(selected["track_id"]) == 1
