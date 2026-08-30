from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import select_radar_measurement_rows


def test_truth_gate_keeps_representable_extreme_spatial_error() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [8.0e307],
            "north_m": [6.0e307],
            "up_m": [0.0],
            "track_id": [17],
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

    with np.errstate(all="raise"):
        selected = select_radar_measurement_rows(
            radar,
            selection="truth-gated",
            truth=truth,
            truth_gate_m=1.0e308,
            truth_time_gate_s=0.0,
        )

    assert selected["track_id"].tolist() == [17]


def test_truth_gate_ignores_overflowing_far_timestamp_for_exact_match() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [1.0e308],
            "east_m": [5.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "track_id": [23],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [-1.0e308, 1.0e308],
            "east_m": [100.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with np.errstate(all="raise"):
        selected = select_radar_measurement_rows(
            radar,
            selection="truth-gated",
            truth=truth,
            truth_gate_m=5.0,
            truth_time_gate_s=0.0,
        )

    assert selected["track_id"].tolist() == [23]


def test_truth_gate_rejects_unrepresentable_spatial_error_without_raising() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [1.0e308],
            "north_m": [0.0],
            "up_m": [0.0],
            "track_id": [31],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [-1.0e308],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    with np.errstate(all="raise"):
        selected = select_radar_measurement_rows(
            radar,
            selection="truth-gated",
            truth=truth,
            truth_gate_m=1.0e308,
            truth_time_gate_s=0.0,
        )

    assert selected.empty
