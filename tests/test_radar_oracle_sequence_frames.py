from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import nearest_candidate_oracle
from raft_uav.evaluation.radar_oracle_diagnostics import time_offset_sweep


def test_oracle_scopes_reused_frame_indices_and_truth_by_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "frame_index": [0, 0],
            "track_id": [11, 22],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
            "up_m": [5.0, 5.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 10.0],
            "up_m": [5.0, 5.0],
        }
    )

    selected = nearest_candidate_oracle(radar, truth, max_time_delta_s=0.1)
    sweep = time_offset_sweep(
        radar,
        truth,
        offsets_s=[0.0],
        max_time_delta_s=0.1,
    )

    assert selected["sequence_id"].tolist() == ["seqA", "seqB"]
    assert selected["track_id"].tolist() == [11, 22]
    assert selected["oracle_candidate_rows"].tolist() == [1, 1]
    np.testing.assert_allclose(selected["oracle_error_3d_m"], [0.0, 0.0])
    assert sweep.loc[0, "count"] == 2.0
    assert sweep.loc[0, "coverage"] == 1.0
    assert sweep.loc[0, "mean_3d_error_m"] == 0.0


def test_oracle_uses_time_keys_when_frame_indices_are_incomplete() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, np.nan],
            "track_id": [1, 2],
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 10.0],
            "north_m": [0.0, 0.0],
            "up_m": [5.0, 5.0],
        }
    )

    selected = nearest_candidate_oracle(radar, truth, max_time_delta_s=0.1)

    assert selected["time_s"].tolist() == [0.0, 1.0]
    assert selected["track_id"].tolist() == [1.0, 2.0]
    assert selected["oracle_candidate_rows"].tolist() == [1.0, 1.0]
    np.testing.assert_allclose(selected["oracle_error_3d_m"], [0.0, 0.0])
