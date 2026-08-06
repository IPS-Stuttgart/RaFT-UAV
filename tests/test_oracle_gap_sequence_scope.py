from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
)


def _pooled_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return radar, truth, radar.copy(), truth.copy()


def test_oracle_gap_scopes_pooled_sequence_frames_and_context() -> None:
    radar, truth, selected, estimates = _pooled_inputs()

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        estimates=estimates,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    ).sort_values("sequence_id")

    assert rows["sequence_id"].tolist() == ["seqA", "seqB"]
    assert rows["candidate_count"].tolist() == [1, 1]
    assert rows["nearest_candidate_track_id"].tolist() == [1, 2]
    assert rows["selected_track_id"].tolist() == [1, 2]
    assert rows["category"].tolist() == [
        "correct_candidate_selected",
        "correct_candidate_selected",
    ]
    np.testing.assert_allclose(rows["nearest_candidate_error_m"], [0.0, 0.0])
    np.testing.assert_allclose(rows["selected_error_m"], [0.0, 0.0])
    np.testing.assert_allclose(rows["estimate_error_m"], [0.0, 0.0])


def test_oracle_gap_does_not_borrow_truth_from_another_sequence() -> None:
    radar, truth, selected, estimates = _pooled_inputs()
    truth = truth.loc[truth["sequence_id"].eq("seqA")].copy()

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=selected,
        estimates=estimates,
    )
    missing = rows.loc[rows["sequence_id"].eq("seqB")].iloc[0]

    assert not bool(missing["truth_available"])
    assert missing["category"] == "no_truth"
    assert np.isnan(missing["nearest_candidate_error_m"])
    assert not bool(missing["selected_present"])
    assert np.isnan(missing["estimate_error_m"])
