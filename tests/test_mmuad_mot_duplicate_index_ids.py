from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0", "seq0"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [5.0, 5.0],
            "track_id": ["uav_1", "uav_1"],
        }
    )


def _anonymous_estimates(index: list[int], id_column: str | None) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "sequence_id": ["seq0", "seq0"],
        "time_s": [0.0, 1.0],
        "state_x_m": [0.0, 1.0],
        "state_y_m": [0.0, 0.0],
        "state_z_m": [5.0, 5.0],
    }
    if id_column is not None:
        data[id_column] = [None, None]
    return pd.DataFrame(data, index=index)


@pytest.mark.parametrize("id_column", [None, "track_id", "output_track_id"])
def test_anonymous_mot_metrics_are_invariant_to_duplicate_index_labels(
    id_column: str | None,
) -> None:
    baseline = compute_multi_object_metrics(
        _anonymous_estimates([0, 1], id_column),
        _truth(),
        match_distance_m=1.0,
    )
    duplicate_index = compute_multi_object_metrics(
        _anonymous_estimates([7, 7], id_column),
        _truth(),
        match_distance_m=1.0,
    )

    assert duplicate_index == baseline
    assert duplicate_index["track_count"] == 2
    assert duplicate_index["id_switches"] == 1
