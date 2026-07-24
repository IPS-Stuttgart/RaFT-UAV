from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.radar_oracle_diagnostics import nearest_candidate_oracle
from raft_uav.evaluation.radar_oracle_diagnostics import time_offset_sweep


def _radar_with_reused_frame_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "frame_index": [7, 7],
            "track_id": [101, 202],
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )


def _matching_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 10.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )


def test_oracle_keeps_reused_frame_indices_separate_by_timestamp() -> None:
    selected = nearest_candidate_oracle(
        _radar_with_reused_frame_index(),
        _matching_truth(),
        max_time_delta_s=0.1,
    )

    assert selected["time_s"].tolist() == [0.0, 10.0]
    assert selected["track_id"].tolist() == [101, 202]
    assert selected["oracle_candidate_rows"].tolist() == [1, 1]
    assert selected["oracle_error_3d_m"].tolist() == pytest.approx([0.0, 0.0])


def test_time_offset_sweep_counts_every_reused_index_frame() -> None:
    sweep = time_offset_sweep(
        _radar_with_reused_frame_index(),
        _matching_truth(),
        offsets_s=[0.0],
        max_time_delta_s=0.1,
    )

    assert sweep.loc[0, "count"] == 2.0
    assert sweep.loc[0, "coverage"] == 1.0
    assert sweep.loc[0, "mean_3d_error_m"] == pytest.approx(0.0)
