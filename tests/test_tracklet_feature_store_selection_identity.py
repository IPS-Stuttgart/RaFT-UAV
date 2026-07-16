from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.tracklet_feature_store import (
    _selection_mask,
    build_tracklet_candidate_feature_store,
)


def _radar() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_index": [0, 1],
            "track_id": [10, 20],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.2, 0.9],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


@pytest.mark.parametrize("selected_track_index", [None, 999])
def test_selected_radar_uses_stable_track_id_across_run_schemas(
    selected_track_index: int | None,
) -> None:
    radar = _radar()
    selected = radar.iloc[[1]].copy()
    if selected_track_index is None:
        selected = selected.drop(columns=["track_index"])
    else:
        selected["track_index"] = selected_track_index

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=_truth(),
        selected_radar=selected,
        truth_time_gate_s=1.0,
    )

    assert features["chosen_by_selected_radar"].tolist() == [False, True]


def test_selected_radar_falls_back_to_track_index_without_track_id() -> None:
    radar = _radar()
    selected = radar.iloc[[1]].drop(columns=["track_id"])

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=_truth(),
        selected_radar=selected,
        truth_time_gate_s=1.0,
    )

    assert features["chosen_by_selected_radar"].tolist() == [False, True]


def test_conflicting_track_id_does_not_match_only_by_track_index() -> None:
    radar = _radar()
    selected = radar.iloc[[1]].copy()
    selected["track_id"] = 999

    features = build_tracklet_candidate_feature_store(
        radar=radar,
        truth=_truth(),
        selected_radar=selected,
        truth_time_gate_s=1.0,
    )

    assert not features["chosen_by_selected_radar"].any()


def test_unidentified_selected_rows_do_not_select_every_candidate() -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index", "frame_index"],
            "frame_key": ["0", "0"],
        }
    )
    selected = pd.DataFrame({"frame_index": [0]})

    assert _selection_mask(features, selected).tolist() == [False, False]
