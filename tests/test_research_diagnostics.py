from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.diagnostics import (
    association_regret,
    candidate_set_recall,
    leakage_sentinel,
    track_switch_metrics,
)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_index": [0, 0, 1, 1],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "east_m": [1.0, 100.0, 9.0, 80.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0, 0.0],
            "track_id": [1, 2, 1, 3],
            "cat_prob_uav": [0.9, 0.1, 0.8, 0.2],
        }
    )


def test_candidate_set_recall_reports_present_target() -> None:
    recall = candidate_set_recall(_radar(), _truth(), distance_gate_m=5.0)
    assert len(recall) == 2
    assert recall["target_present"].all()
    assert np.isclose(recall["best_candidate_error_m"].max(), 1.0)


def test_association_regret_and_switch_metrics() -> None:
    selected = _radar().iloc[[1, 2]].copy()
    regret = association_regret(selected, _radar(), _truth())
    assert len(regret) == 2
    assert regret["association_regret_m"].iloc[0] > 50.0
    metrics = track_switch_metrics(selected)
    assert metrics["track_switch_count"] == 1
    assert metrics["unique_track_ids"] == 2


def test_association_diagnostics_handle_unsorted_truth_times() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [1.0, 0.0],
            "east_m": [10.0, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )
    radar = pd.DataFrame(
        {
            "frame_index": [0, 1],
            "time_s": [0.05, 1.05],
            "east_m": [0.5, 10.5],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "track_id": [1, 2],
        }
    )

    recall = candidate_set_recall(
        radar,
        truth,
        distance_gate_m=1.0,
        max_time_delta_s=0.2,
    )
    regret = association_regret(radar, radar, truth, max_time_delta_s=0.2)

    assert recall["target_present"].tolist() == [True, True]
    np.testing.assert_allclose(regret["selected_error_m"], [0.5, 0.5])


def test_leakage_sentinel_flags_training_reference() -> None:
    payload = {"training_flights": ["Opt1", "Opt3"], "heldout_flight": "Opt3"}
    violations = leakage_sentinel(payload, heldout_flight="Opt3")
    assert violations
    assert violations[0].path == "training_flights[1]"
