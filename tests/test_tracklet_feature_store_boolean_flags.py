from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
    summarize_counterfactual_regret,
)


def test_dashboard_parses_persisted_candidate_selection_flags() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index", "frame_index"],
            "frame_key": ["7", "7"],
            "time_s": [1.0, 1.0],
            "track_id": [11, 22],
            "track_index": [0, 1],
            "oracle_error_m": [1.0, 5.0],
            "oracle_rank_in_frame": [1.0, 2.0],
            "chosen_by_selected_radar": ["False", "True"],
        }
    )

    dashboard = build_counterfactual_association_dashboard(features)
    row = dashboard.iloc[0]

    assert row["selected_candidate_track_id"] == 22
    assert row["selected_candidate_error_m"] == pytest.approx(5.0)
    assert row["selection_regret_m"] == pytest.approx(4.0)
    assert row["category"] == "wrong_candidate_selected"


def test_regret_summary_parses_persisted_availability_flags() -> None:
    regret = pd.DataFrame(
        {
            "truth_available": ["False", "True", "0", "1"],
            "selected_present": ["True", "False", "1", "0"],
            "selection_regret_m": [100.0, 200.0, 3.0, 4.0],
            "category": ["a", "b", "c", "d"],
        }
    )

    summary = summarize_counterfactual_regret(regret)

    assert summary["radar_frame_count"] == 4
    assert summary["truth_matched_frame_count"] == 2
    assert summary["selected_frame_count"] == 0


def test_dashboard_rejects_ambiguous_selection_flags() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index"],
            "frame_key": ["7"],
            "time_s": [1.0],
            "oracle_error_m": [1.0],
            "oracle_rank_in_frame": [1.0],
            "chosen_by_selected_radar": ["maybe"],
        },
        index=[12],
    )

    with pytest.raises(
        ValueError,
        match=r"chosen_by_selected_radar contains invalid Boolean values at rows \[12\]",
    ):
        build_counterfactual_association_dashboard(features)
