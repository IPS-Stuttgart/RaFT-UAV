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


def _candidate_rows_with_duplicate_bootstrap(*, reverse: bool) -> pd.DataFrame:
    first = {
        "time_s": 0.0,
        "source": "radar",
        "track_id": "selected",
        "x_m": 0.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "std_xy_m": 1.0,
        "std_z_m": 1.0,
    }
    second = {**first, "x_m": 2.0}
    anchor = {
        "time_s": 0.0,
        "source": "camera",
        "track_id": "anchor",
        "x_m": 4.0,
        "y_m": 0.0,
        "z_m": 0.0,
        "std_xy_m": 1.0,
        "std_z_m": 1.0,
    }
    later = {**first, "time_s": 1.0, "x_m": 1.0}
    bootstrap_rows = (second, first) if reverse else (first, second)
    return pd.DataFrame.from_records([*bootstrap_rows, anchor, later])


def _tracker_config() -> TrackerConfig:
    return TrackerConfig(
        acceleration_std_mps2=1.0,
        primary_covariance_scale=1.0,
        secondary_covariance_scale=1.0,
        soft_anchor_cap_m=1.0,
        soft_anchor_gate_m=10.0,
    )


def test_same_timestamp_filter_updates_ignore_input_row_order() -> None:
    candidates_a = _candidate_rows(selected_first=True)
    candidates_b = _candidate_rows(selected_first=False)
    selected = candidates_a.loc[candidates_a["track_id"] == "selected"].copy()

    result_a = _run_sequence_filter(
        candidates_a,
        selected,
        sequence_truth=None,
        config=_tracker_config(),
    )
    result_b = _run_sequence_filter(
        candidates_b,
        selected,
        sequence_truth=None,
        config=_tracker_config(),
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


def test_duplicate_selected_bootstrap_ignores_input_row_order() -> None:
    candidates_a = _candidate_rows_with_duplicate_bootstrap(reverse=False)
    candidates_b = _candidate_rows_with_duplicate_bootstrap(reverse=True)
    selected_a = candidates_a.loc[candidates_a["track_id"] == "selected"].copy()
    selected_b = candidates_b.loc[candidates_b["track_id"] == "selected"].copy()

    result_a = _run_sequence_filter(
        candidates_a,
        selected_a,
        sequence_truth=None,
        config=_tracker_config(),
    )
    result_b = _run_sequence_filter(
        candidates_b,
        selected_b,
        sequence_truth=None,
        config=_tracker_config(),
    )

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
