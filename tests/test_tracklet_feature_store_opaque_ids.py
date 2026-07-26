from __future__ import annotations

import pandas as pd

from raft_uav.diagnostics.tracklet_feature_store import _selection_mask


def _features(track_ids: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_key_type": ["frame_index"] * len(track_ids),
            "frame_key": ["0"] * len(track_ids),
            "track_id": track_ids,
            "track_index": list(range(len(track_ids))),
        }
    )


def test_selection_mask_matches_opaque_track_ids_without_track_index() -> None:
    features = _features(["uav-A", "uav-B"])
    selected = pd.DataFrame({"frame_index": [0], "track_id": ["uav-B"]})

    assert _selection_mask(features, selected).tolist() == [False, True]


def test_selection_mask_does_not_truncate_fractional_track_ids() -> None:
    features = _features([1, 1.5])
    selected = pd.DataFrame({"frame_index": [0], "track_id": [1.5]})

    assert _selection_mask(features, selected).tolist() == [False, True]
