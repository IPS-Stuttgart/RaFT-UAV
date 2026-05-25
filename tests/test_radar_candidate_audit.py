from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.radar_candidate_audit import (
    EvaluationWindow,
    build_candidate_residual_frame,
    parse_clock_delta_values,
    parse_evaluation_window,
    select_oracle_candidate_per_frame,
    summarize_oracle_selection,
)


def test_oracle_candidate_selects_lowest_error_per_frame() -> None:
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 10,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
                "cat_prob_uav": 0.9,
            },
            {
                "frame_index": 1,
                "track_id": 11,
                "time_s": 0.0,
                "east_m": 0.5,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 0.5,
                "cat_prob_uav": 0.1,
            },
            {
                "frame_index": 2,
                "track_id": 12,
                "time_s": 1.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
                "cat_prob_uav": 0.8,
            },
        ]
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    residuals = build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="fortem-lla",
        radar_clock_delta_s=0.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=0.1,
    )
    selected = select_oracle_candidate_per_frame(residuals)

    assert selected["track_id"].astype(int).tolist() == [11, 12]
    assert np.allclose(selected["error_3d_m"].to_numpy(), [0.5, 0.0])


def test_clock_delta_changes_truth_alignment() -> None:
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "track_id": 1,
                "time_s": 5.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
            }
        ]
    )
    truth = pd.DataFrame(
        {
            "time_s": [5.0, 6.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    unshifted = build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="fortem-lla",
        radar_clock_delta_s=0.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=2.0,
    )
    shifted = build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="fortem-lla",
        radar_clock_delta_s=1.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=2.0,
    )

    assert float(unshifted["error_3d_m"].iloc[0]) == 10.0
    assert float(shifted["error_3d_m"].iloc[0]) == 0.0


def test_polar_position_source_uses_azimuth_convention() -> None:
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 10.0,
                "azimuth_deg": 90.0,
                "elevation_deg": 0.0,
            }
        ]
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [10.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    residuals = build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="polar-from-lw1",
        radar_clock_delta_s=0.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=0.1,
        azimuth_convention="north-clockwise",
        elevation_mode="as-is",
    )

    assert np.isclose(float(residuals["candidate_east_m"].iloc[0]), 10.0)
    assert np.isclose(float(residuals["error_3d_m"].iloc[0]), 0.0)


def test_summary_reports_range_gate_recall_and_counts() -> None:
    radar = pd.DataFrame(
        [
            {
                "frame_index": 1,
                "time_s": 0.0,
                "east_m": 0.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 700.0,
                "cat_prob_uav": 0.9,
            }
        ]
    )
    truth = pd.DataFrame(
        {"time_s": [0.0], "east_m": [0.0], "north_m": [0.0], "up_m": [0.0]}
    )
    residuals = build_candidate_residual_frame(
        radar=radar,
        truth=truth,
        position_source="fortem-lla",
        radar_clock_delta_s=0.0,
        evaluation_window=EvaluationWindow("truth-window"),
        range_gate_m=800.0,
        max_truth_time_delta_s=0.1,
    )
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

    assert summary["count_frames"] == 1
    assert summary["mean_3d_m"] == 0.0
    assert summary["frames_oracle_range_le_gate"] == 1
    assert summary["recall_3d_le_25m"] == 1.0


def test_parse_clock_delta_values_and_explicit_window() -> None:
    assert parse_clock_delta_values(["-1,0"], ["1:3:1"]) == [-1.0, 0.0, 1.0, 2.0, 3.0]
    window = parse_evaluation_window("explicit:1.5:3.0")

    assert np.array_equal(window.contains(np.array([1.0, 2.0, 4.0])), [False, True, False])
