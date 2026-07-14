from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
    selected_track_stability_metrics,
)


def test_oracle_gap_decomposition_scopes_pooled_sequences() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
            "up_m": [0.0, 0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
            "up_m": [0.0, 0.0],
        }
    )
    selected = radar.copy()
    estimates = truth.copy()

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        estimates=estimates,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    ).sort_values("sequence_id")

    assert rows["sequence_id"].tolist() == ["seqA", "seqB"]
    assert rows["candidate_count"].tolist() == [1, 1]
    assert rows["category"].tolist() == [
        "correct_candidate_selected",
        "correct_candidate_selected",
    ]
    np.testing.assert_allclose(rows["nearest_candidate_error_m"], [0.0, 0.0])
    np.testing.assert_allclose(rows["selected_error_m"], [0.0, 0.0])
    np.testing.assert_allclose(rows["estimate_error_m"], [0.0, 0.0])


def test_oracle_gap_does_not_borrow_truth_from_another_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqMissing"],
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": [7],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    rows = decompose_radar_oracle_gap(radar=radar, truth=truth)

    assert rows.loc[0, "sequence_id"] == "seqMissing"
    assert not bool(rows.loc[0, "truth_available"])
    assert rows.loc[0, "category"] == "no_truth"
    assert np.isnan(rows.loc[0, "nearest_candidate_error_m"])


def test_track_stability_ignores_sequence_boundaries() -> None:
    selected = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "track_id": [1, 1, 2, 2],
        }
    )

    metrics = selected_track_stability_metrics(selected)

    assert metrics["selected_sequence_count"] == 2
    assert metrics["track_switch_count"] == 0
    assert metrics["track_switch_rate"] == 0.0
    assert metrics["selected_time_gap_max_s"] == 1.0
