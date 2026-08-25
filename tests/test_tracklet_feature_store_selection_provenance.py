from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import raft_uav.diagnostics.tracklet_feature_store as feature_store


def test_counterfactual_dashboard_preserves_opaque_candidate_identifiers() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index", "frame_index"],
            "frame_key": ["0", "0"],
            "time_s": [0.0, 0.0],
            "oracle_rank_in_frame": [1.0, 2.0],
            "oracle_error_m": [1.0, 2.0],
            "chosen_by_selected_radar": [False, True],
            "track_id": ["uav-A", "uav-B"],
            "track_index": [0.5, 1.5],
        }
    )

    dashboard = feature_store.build_counterfactual_association_dashboard(features)

    assert dashboard.loc[0, "best_candidate_track_id"] == "uav-A"
    assert dashboard.loc[0, "selected_candidate_track_id"] == "uav-B"
    assert dashboard.loc[0, "best_candidate_track_index"] == 0.5
    assert dashboard.loc[0, "selected_candidate_track_index"] == 1.5


def test_counterfactual_dashboard_preserves_zero_padded_track_id() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index"],
            "frame_key": ["0"],
            "time_s": [0.0],
            "oracle_rank_in_frame": [1.0],
            "oracle_error_m": [1.0],
            "chosen_by_selected_radar": [True],
            "track_id": ["001"],
            "track_index": [0],
        }
    )

    dashboard = feature_store.build_counterfactual_association_dashboard(features)

    assert dashboard.loc[0, "best_candidate_track_id"] == "001"
    assert dashboard.loc[0, "selected_candidate_track_id"] == "001"


def test_external_selected_radar_rejects_multi_flight_reuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be applied to multiple flights"):
        feature_store._validate_external_selected_radar_scope(
            tmp_path / "selected_radar.csv",
            ["flight-a", "flight-b"],
        )


def test_external_selected_radar_allows_single_flight(tmp_path: Path) -> None:
    feature_store._validate_external_selected_radar_scope(
        tmp_path / "selected_radar.csv",
        ["flight-a"],
    )
