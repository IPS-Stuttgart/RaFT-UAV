from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import drop_radar_frames


class _AlternatingRng:
    def random(self, size: int) -> np.ndarray:
        assert size == 2
        return np.asarray([0.0, 1.0])


def test_radar_dropout_samples_equal_frame_indices_per_sequence() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB"],
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [1, 2],
        }
    )

    retained = drop_radar_frames(
        radar,
        rate=0.5,
        rng=_AlternatingRng(),  # type: ignore[arg-type]
    )

    assert retained["sequence_id"].tolist() == ["seqB"]
