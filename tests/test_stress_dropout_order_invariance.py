from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import drop_radar_frames, drop_rf_bursts


def _keys(frame: pd.DataFrame, columns: list[str]) -> set[tuple[object, ...]]:
    return set(frame.loc[:, columns].itertuples(index=False, name=None))


def test_seeded_radar_dropout_is_invariant_to_input_row_order() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["b", "a", "b", "a", "b", "a"],
            "frame_index": [2, 0, 0, 2, 1, 1],
            "time_s": [2.0, 0.0, 0.0, 2.0, 1.0, 1.0],
        }
    )
    shuffled = radar.sample(frac=1.0, random_state=7).reset_index(drop=True)

    retained = drop_radar_frames(radar, rate=0.5, rng=np.random.default_rng(0))
    retained_shuffled = drop_radar_frames(
        shuffled,
        rate=0.5,
        rng=np.random.default_rng(0),
    )

    assert _keys(retained, ["sequence_id", "frame_index"]) == _keys(
        retained_shuffled,
        ["sequence_id", "frame_index"],
    )


def test_seeded_rf_burst_dropout_is_invariant_to_input_row_order() -> None:
    rf = pd.DataFrame(
        {
            "sequence_id": ["b", "a", "b", "a", "b", "a"],
            "time_s": [12.0, 0.0, 0.0, 12.0, 6.0, 6.0],
        }
    )
    shuffled = rf.sample(frac=1.0, random_state=7).reset_index(drop=True)

    retained = drop_rf_bursts(rf, rate=0.5, rng=np.random.default_rng(0))
    retained_shuffled = drop_rf_bursts(
        shuffled,
        rate=0.5,
        rng=np.random.default_rng(0),
    )

    assert _keys(retained, ["sequence_id", "time_s"]) == _keys(
        retained_shuffled,
        ["sequence_id", "time_s"],
    )
