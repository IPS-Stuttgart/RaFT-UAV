from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import drop_rf_bursts


class _DeterministicRng:
    def random(self, size: int) -> np.ndarray:
        return np.array([0.75, 0.25], dtype=float)[:size]


def test_rf_burst_dropout_is_scoped_by_sequence() -> None:
    rf = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "time_s": [0.0, 0.0],
            "east_m": [1.0, 2.0],
        }
    )

    out = drop_rf_bursts(rf, rate=0.5, rng=_DeterministicRng())

    assert out["sequence_id"].tolist() == ["seq-a"]
    assert out["east_m"].tolist() == [1.0]
