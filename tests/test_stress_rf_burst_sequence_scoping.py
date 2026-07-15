from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import drop_rf_bursts


class _AlternatingRng:
    def __init__(self) -> None:
        self._call = 0

    def random(self, size: int) -> np.ndarray:
        assert size == 1
        values = np.asarray([0.0, 1.0])
        value = values[self._call]
        self._call += 1
        return np.asarray([value])


def test_rf_burst_sampling_is_scoped_by_sequence() -> None:
    rf = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "east_m": [0.0, 1.0, 100.0, 101.0],
        }
    )

    retained = drop_rf_bursts(
        rf,
        rate=0.5,
        rng=_AlternatingRng(),  # type: ignore[arg-type]
    )

    assert retained["sequence_id"].tolist() == ["seqB", "seqB"]
    assert retained["east_m"].tolist() == [100.0, 101.0]
