from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.radar_candidate_audit import (
    EvaluationWindow,
    build_candidate_residual_frame,
    select_oracle_candidate_per_frame,
    summarize_oracle_selection,
)


def _audit_rows(radar: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="fortem-lla",
        radar_clock_delta_s=0.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=0.1,
    )


def test_missing_frame_indices_fall_back_to_time() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [np.nan, np.nan],
            "track_id": [10, 11],
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "range_m": [1.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    residuals = _audit_rows(radar, truth)
    selected = select_oracle_candidate_per_frame(residuals)
    summary = summarize_oracle_selection(
        selected,
        residuals,
        position_source="fortem-lla",
        radar_clock_delta_s=0.0,
        azimuth_convention="not-applicable",
        elevation_mode="not-applicable",
        range_gate_m=800.0,
    )

    assert selected["track_id"].astype(int).tolist() == [10, 11]
    assert summary["count_frames"] == 2


def test_reused_frame_counter_at_new_time_remains_separate() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [7, 7],
            "track_id": [20, 21],
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "range_m": [1.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    selected = select_oracle_candidate_per_frame(_audit_rows(radar, truth))

    assert selected["track_id"].astype(int).tolist() == [20, 21]


def test_missing_index_candidate_joins_unique_indexed_frame() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [3.0, np.nan],
            "track_id": [30, 31],
            "time_s": [0.0, 0.0],
            "east_m": [5.0, 0.5],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "range_m": [5.0, 0.5],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    selected = select_oracle_candidate_per_frame(_audit_rows(radar, truth))

    assert selected["track_id"].astype(int).tolist() == [31]
