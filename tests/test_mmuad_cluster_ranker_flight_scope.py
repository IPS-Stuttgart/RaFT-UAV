from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad import cluster_ranker


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "source": ["lidar", "lidar"],
            "track_id": ["a", "b"],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "x_m": [0.0, 100.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def test_cluster_ranker_truth_labels_stay_within_physical_flight() -> None:
    labeled = cluster_ranker.label_cluster_features_against_truth(
        _features(),
        _truth().iloc[::-1].reset_index(drop=True),
        max_truth_time_delta_s=0.0,
    )

    assert labeled["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert labeled["truth_distance_3d_m"].tolist() == [0.0, 0.0]
    assert labeled["truth_matched"].tolist() == [True, True]
    assert labeled["good_cluster"].tolist() == [True, True]


def test_cluster_ranker_rejects_ambiguous_one_sided_flight_scope() -> None:
    with pytest.raises(ValueError, match="ambiguous flight_id metadata"):
        cluster_ranker.label_cluster_features_against_truth(
            _features(),
            _truth().drop(columns="flight_id"),
        )


def test_cluster_ranker_allows_unambiguous_one_sided_flight_scope() -> None:
    labeled = cluster_ranker.label_cluster_features_against_truth(
        _features().iloc[[0]].copy(),
        _truth().iloc[[0]].drop(columns="flight_id"),
    )

    assert labeled.loc[labeled.index[0], "truth_distance_3d_m"] == 0.0


def test_cluster_ranker_rejects_partial_flight_scope() -> None:
    features = _features()
    features.loc[1, "flight_id"] = None

    with pytest.raises(ValueError, match="partially missing flight_id"):
        cluster_ranker.label_cluster_features_against_truth(features, _truth())


def test_cluster_ranker_frame_diagnostics_stay_flight_local() -> None:
    labeled = cluster_ranker.label_cluster_features_against_truth(_features(), _truth())
    labeled["ranker_score"] = [0.2, 0.8]

    frames = cluster_ranker._ranker_frame_selection_rows(labeled)

    assert frames[["sequence_id", "flight_id"]].values.tolist() == [
        ["shared", "flight-a"],
        ["shared", "flight-b"],
    ]
    assert frames["selected_truth_distance_3d_m"].tolist() == [0.0, 0.0]
