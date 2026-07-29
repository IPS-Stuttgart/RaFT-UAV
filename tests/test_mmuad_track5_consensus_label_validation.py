from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    build_track5_consensus_estimate_ensemble,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})


def _estimate(x: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x],
            "state_y_m": [0.0],
            "state_z_m": [1.0],
        }
    )


def test_consensus_ensemble_rejects_duplicate_input_labels() -> None:
    with pytest.raises(ValueError, match="estimate input label 'same' is duplicated"):
        build_track5_consensus_estimate_ensemble(
            [("same", _estimate(0.0), 1.0), ("same", _estimate(1.0), 1.0)],
            _template(),
        )


def test_consensus_ensemble_rejects_normalized_label_collisions() -> None:
    with pytest.raises(ValueError, match="collide after normalization"):
        build_track5_consensus_estimate_ensemble(
            [("near branch", _estimate(0.0), 1.0), ("near_branch", _estimate(1.0), 1.0)],
            _template(),
        )
