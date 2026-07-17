from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import association_regret
from raft_uav.research.diagnostics import track_switch_metrics


def test_association_regret_keeps_fractional_frame_indices_distinct() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    radar = pd.DataFrame(
        {
            "frame_index": [1.25, 1.75],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "track_id": [10, 20],
        }
    )

    regret = association_regret(
        radar.iloc[[0]].copy(),
        radar,
        truth,
        max_time_delta_s=0.1,
    )

    assert regret.loc[0, "event_key"] == "frame_index:1.25"
    assert regret.loc[0, "candidate_count"] == 1
    assert regret.loc[0, "selected_track_id"] == 10
    assert regret.loc[0, "best_track_id"] == 10
    assert regret.loc[0, "association_regret_m"] == 0.0


def test_association_regret_does_not_truncate_fractional_track_ids() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    radar = pd.DataFrame(
        {
            "frame_index": [0],
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
            "track_id": [12.75],
        }
    )

    regret = association_regret(radar, radar, truth, max_time_delta_s=0.1)

    assert regret.loc[0, "selected_track_id"] is None
    assert regret.loc[0, "best_track_id"] is None


def test_track_switch_metrics_ignore_malformed_track_identifiers() -> None:
    selected = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "track_id": [1.25, np.bool_(True), "2.5"],
        }
    )

    metrics = track_switch_metrics(selected)

    assert metrics["selected_radar_rows"] == 3
    assert metrics["track_switch_count"] == 0
    assert metrics["unique_track_ids"] == 0
    assert np.isnan(metrics["dominant_track_fraction"])


def test_track_switch_metrics_keep_exact_integer_like_identifiers() -> None:
    selected = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "track_id": ["7.0", np.int64(7), 8.0],
        }
    )

    metrics = track_switch_metrics(selected)

    assert metrics["track_switch_count"] == 1
    assert metrics["unique_track_ids"] == 2
