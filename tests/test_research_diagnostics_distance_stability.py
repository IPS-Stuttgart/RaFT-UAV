from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research import diagnostics as diagnostics_module
from raft_uav.research.diagnostics import association_regret, candidate_set_recall


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


def test_candidate_recall_keeps_large_representable_distance_finite() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0],
            "time_s": [0.0],
            "east_m": [6.0e307],
            "north_m": [8.0e307],
            "up_m": [0.0],
            "track_id": [7],
        }
    )

    with np.errstate(over="raise", invalid="raise"):
        recall = candidate_set_recall(
            radar,
            _truth(),
            distance_gate_m=1.01e308,
            max_time_delta_s=0.1,
        )

    assert recall.loc[0, "best_candidate_error_m"] == pytest.approx(1.0e308)
    assert bool(recall.loc[0, "target_present"])


def test_association_regret_ranks_large_representable_distances_correctly() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0],
            "time_s": [0.0, 0.0],
            "east_m": [9.0e307, 6.0e307],
            "north_m": [9.0e307, 8.0e307],
            "up_m": [0.0, 0.0],
            "track_id": [11, 22],
        }
    )
    selected = radar.iloc[[0]].copy()

    with np.errstate(over="raise", invalid="raise"):
        regret = association_regret(
            selected,
            radar,
            _truth(),
            max_time_delta_s=0.1,
        )

    selected_error = float(np.hypot(9.0e307, 9.0e307))
    best_error = 1.0e308
    assert regret.loc[0, "selected_error_m"] == pytest.approx(selected_error)
    assert regret.loc[0, "best_candidate_error_m"] == pytest.approx(best_error)
    assert regret.loc[0, "best_track_id"] == 22
    assert regret.loc[0, "association_regret_m"] == pytest.approx(
        selected_error - best_error
    )


def test_stable_norm_repair_preserves_keepdims_shape() -> None:
    values = np.array(
        [
            [6.0e307, 8.0e307, 0.0],
            [3.0, 4.0, 0.0],
        ]
    )

    with np.errstate(over="raise", invalid="raise"):
        result = diagnostics_module._LEGACY.np.linalg.norm(
            values,
            axis=1,
            keepdims=True,
        )

    assert result.shape == (2, 1)
    assert result[:, 0] == pytest.approx([1.0e308, 5.0])
