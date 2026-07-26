from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.tracker import TrackerConfig, _run_sequence_filter


def _candidate_rows(*, low_uncertainty_first: bool) -> pd.DataFrame:
    low_uncertainty_anchor = {
        "time_s": 0.0,
        "source": "camera",
        "track_id": "anchor",
        "x_m": 10.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "std_xy_m": 0.5,
        "std_z_m": 0.5,
    }
    high_uncertainty_anchor = {
        "time_s": 0.0,
        "source": "camera",
        "track_id": "anchor",
        "x_m": 10.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "std_xy_m": 20.0,
        "std_z_m": 20.0,
    }
    selected = {
        "time_s": 0.0,
        "source": "radar",
        "track_id": "selected",
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "std_xy_m": 1.0,
        "std_z_m": 1.0,
    }
    anchors = (
        (low_uncertainty_anchor, high_uncertainty_anchor)
        if low_uncertainty_first
        else (high_uncertainty_anchor, low_uncertainty_anchor)
    )
    return pd.DataFrame.from_records((*anchors, selected))


def test_same_timestamp_covariance_ties_ignore_input_row_order() -> None:
    candidates_a = _candidate_rows(low_uncertainty_first=True)
    candidates_b = _candidate_rows(low_uncertainty_first=False)
    selected = candidates_a.loc[candidates_a["track_id"] == "selected"].copy()
    config = TrackerConfig(
        acceleration_std_mps2=1.0,
        primary_covariance_scale=1.0,
        secondary_covariance_scale=1.0,
        soft_anchor_cap_m=1.0,
        soft_anchor_gate_m=20.0,
    )

    result_a = _run_sequence_filter(
        candidates_a,
        selected,
        sequence_truth=None,
        config=config,
    )
    result_b = _run_sequence_filter(
        candidates_b,
        selected,
        sequence_truth=None,
        config=config,
    )

    assert result_a["update_action"].tolist() == [
        "soft_anchor",
        "soft_anchor",
        "selected_update",
    ]
    np.testing.assert_allclose(
        result_a[
            [
                "state_x_m",
                "state_y_m",
                "state_z_m",
                "v_x_mps",
                "v_y_mps",
                "v_z_mps",
            ]
        ],
        result_b[
            [
                "state_x_m",
                "state_y_m",
                "state_z_m",
                "v_x_mps",
                "v_y_mps",
                "v_z_mps",
            ]
        ],
        rtol=0.0,
        atol=0.0,
    )
