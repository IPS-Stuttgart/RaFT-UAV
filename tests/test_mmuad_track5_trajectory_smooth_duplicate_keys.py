from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_smooth import smooth_track5_submission_rows


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 20.0, 30.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0, 1.0],
            "Classification": [2, 2, 2, 2],
        }
    )


def test_trajectory_smoother_rejects_duplicate_sequence_timestamp_keys() -> None:
    with pytest.raises(ValueError, match="duplicate sequence/timestamp keys"):
        smooth_track5_submission_rows(
            _rows(),
            window_s=0.25,
            bandwidth_s=0.1,
            min_neighbors=3,
        )


def test_trajectory_smoother_allows_same_timestamp_in_different_sequences() -> None:
    rows = _rows().iloc[[0, 1, 3]].copy()
    second = rows.copy()
    second["sequence_id"] = "seq0002"
    combined = pd.concat([rows, second], ignore_index=True)

    smoothed, diagnostics = smooth_track5_submission_rows(
        combined,
        window_s=0.25,
        bandwidth_s=0.1,
        min_neighbors=3,
    )

    assert len(smoothed) == len(combined)
    assert len(diagnostics) == len(combined)
    assert not smoothed.duplicated(["sequence_id", "time_s"]).any()
