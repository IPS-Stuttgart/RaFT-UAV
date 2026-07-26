from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.tracker import TrackerConfig, _run_sequence_filter


def _candidate_rows(*, selected_first: bool) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_s, selected_x, anchor_x in ((0.0, 0.0, 4.0), (1.0, 1.0, 5.0)):
        selected = {
            "time_s": time_s,
            "source": "radar",
            "track_id": "selected",
            "x_m": selected_x,
            "y_m": 0.0,
            "z_m": 0.0,
            "std_xy_m": 1.0,
            "std_z_m": 1.0,
        }
        anchor = {
            "time_s": time_s,
            "source": "camera",
            "track_id": "anchor",
            "x_m": anchor_x,
            "y_m": 0.0,
            "z_m": 0.0,
            "std_xy_m": 1.0,
            "std_z_m": 1.0,
        }
        records.extend((selected, anchor) if selected_first else (anchor, selected))
    return pd.DataFrame.from_records(records)


def test_same_timestamp_filter_updates_ignore_input_row_order() -> None:
    candidates_a = _candidate_rows(selected_first=True)
    candidates_b = _candidate_rows(selected_first=False)
    selected = candidates_a.loc[candidates_a["track_id"] == "selected"].copy()
    config = TrackerConfig(
        acceleration_std_mps2=1.0,
        primary_covariance_scale=1.0,
        secondary_covariance_scale=1.0,
        soft_anchor_cap_m=1.0,
        soft_anchor_gate_m=10.0,
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
        "selected_update",
        "soft_anchor",
        "selected_update",
    ]
    np.testing.assert_allclose(
        result_a[["state_x_m", "state_y_m", "state_z_m", "v_x_mps"]],
        result_b[["state_x_m", "state_y_m", "state_z_m", "v_x_mps"]],
        rtol=0.0,
        atol=0.0,
    )
