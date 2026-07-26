from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines import tracklet_viterbi as tv


def _node(frame_index: int, east_m: float | None) -> tv._ViterbiNode:
    event_key = ("frame_index", frame_index)
    if east_m is None:
        return tv._ViterbiNode(
            event_index=frame_index,
            event_key=event_key,
            time_s=float(frame_index),
            row=None,
            position=None,
            velocity=None,
            track_id=None,
            unary_cost=0.0,
            anchor_nis=0.0,
            catprob_cost=0.0,
            range_cost=0.0,
            is_miss=True,
        )

    row = pd.Series(
        {
            "frame_index": frame_index,
            "time_s": float(frame_index),
            "east_m": east_m,
            "north_m": 0.0,
            "up_m": 0.0,
        }
    )
    return tv._ViterbiNode(
        event_index=frame_index,
        event_key=event_key,
        time_s=float(frame_index),
        row=row,
        position=np.array([east_m, 0.0, 0.0], dtype=float),
        velocity=None,
        track_id=None,
        unary_cost=0.0,
        anchor_nis=0.0,
        catprob_cost=0.0,
        range_cost=0.0,
    )


def test_soft_viterbi_does_not_resurrect_minority_path_detection() -> None:
    config = tv.TrackletViterbiAssociationConfig(
        soft_top_k_paths=2,
        soft_path_temperature=1.0,
    )
    paths = [
        (0.0, [_node(1, None)]),
        (4.0, [_node(1, 10.0)]),
    ]

    selected = tv._selected_rows_from_soft_viterbi_paths(paths, config)

    assert selected == []


def test_soft_viterbi_reports_detection_and_miss_path_mass() -> None:
    config = tv.TrackletViterbiAssociationConfig(
        soft_top_k_paths=3,
        soft_path_temperature=1.0,
    )
    paths = [
        (0.0, [_node(1, 0.0)]),
        (1.0, [_node(1, 10.0)]),
        (3.0, [_node(1, None)]),
    ]

    selected = tv._selected_rows_from_soft_viterbi_paths(paths, config)

    assert len(selected) == 1
    row = selected[0]
    global_weights = tv._soft_path_weights(np.array([0.0, 1.0, 3.0]), config)
    detection_probability = float(global_weights[0] + global_weights[1])
    conditional_weights = tv._soft_path_weights(np.array([0.0, 1.0]), config)

    assert int(row["association_soft_path_count"]) == 3
    assert int(row["association_soft_detection_path_count"]) == 2
    np.testing.assert_allclose(
        float(row["association_soft_detection_probability"]),
        detection_probability,
    )
    np.testing.assert_allclose(
        float(row["association_soft_miss_probability"]),
        1.0 - detection_probability,
    )
    np.testing.assert_allclose(
        float(row["east_m"]),
        float(conditional_weights @ np.array([0.0, 10.0])),
    )
