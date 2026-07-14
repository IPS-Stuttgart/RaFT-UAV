from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.tracker import add_truth_errors


def test_tracker_truth_errors_are_scoped_by_sequence() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqB", "seqA", "seqMissing"],
            "time_s": [0.0, 0.0, 0.0],
            "state_x_m": [100.0, 0.0, 50.0],
            "state_y_m": [10.0, 0.0, 5.0],
            "state_z_m": [0.0, 0.0, 0.0],
        },
        index=[7, 7, 42],
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 10.0],
            "z_m": [0.0, 0.0],
        }
    )

    scored = add_truth_errors(estimates, truth)

    assert scored.index.tolist() == [7, 7, 42]
    assert scored["sequence_id"].tolist() == ["seqB", "seqA", "seqMissing"]
    np.testing.assert_allclose(scored["truth_x_m"].iloc[:2], [100.0, 0.0])
    np.testing.assert_allclose(scored["error_3d_m"].iloc[:2], [0.0, 0.0])
    assert np.isnan(scored["truth_x_m"].iloc[2])
    assert np.isnan(scored["error_3d_m"].iloc[2])


def test_tracker_truth_errors_preserve_sequence_less_behavior() -> None:
    estimates = pd.DataFrame(
        {
            "time_s": [0.0],
            "state_x_m": [1.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    scored = add_truth_errors(estimates, truth)

    assert scored["truth_x_m"].tolist() == [0.0]
    assert scored["error_3d_m"].tolist() == [1.0]
