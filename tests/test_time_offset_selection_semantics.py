from __future__ import annotations

import numpy as np
import pandas as pd

import raft_uav.diagnostics.time_offset as time_offset


def test_longest_track_counts_distinct_physical_frames() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0, 0, 1, 2],
            "time_s": [0.0, 0.0, 0.0, 1.0, 2.0],
            "track_id": [1, 1, 1, 2, 2],
            "cat_prob_uav": [0.9, 0.8, 0.7, 0.9, 0.9],
            "east_m": [100.0, 200.0, 300.0, 1.0, 2.0],
            "north_m": [0.0] * 5,
            "up_m": [0.0] * 5,
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    assert time_offset._longest_track_id(radar) == 2

    sweep = time_offset.sweep_radar_against_truth(
        radar=radar,
        truth=truth,
        taus_s=[0.0],
        dimensions=2,
        selection="longest-track",
        catprob_threshold=0.4,
        max_truth_time_delta_s=0.1,
    )

    row = sweep.iloc[0]
    assert row["candidate_count"] == 3
    assert row["selected_count"] == 2
    assert row["matched_count"] == 2
    assert row["rmse_error_m"] == 0.0


def test_best_offset_row_ignores_nonfinite_offset_with_better_metric() -> None:
    sweep = pd.DataFrame(
        {
            "tau_s": [np.nan, 1.0],
            "p95_error_m": [0.0, 1.0],
        }
    )

    best = time_offset.best_offset_row(sweep, objective="p95")

    assert float(best["tau_s"]) == 1.0


def test_best_offset_row_prefers_smallest_correction_on_flat_sweep() -> None:
    sweep = pd.DataFrame(
        {
            "tau_s": [2.0, -2.0, 0.0],
            "p95_error_m": [5.0, 5.0, 5.0],
        }
    )

    forward = time_offset.best_offset_row(sweep, objective="p95")
    reverse = time_offset.best_offset_row(
        sweep.iloc[::-1].reset_index(drop=True),
        objective="p95",
    )

    assert float(forward["tau_s"]) == 0.0
    assert float(reverse["tau_s"]) == 0.0
