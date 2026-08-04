from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.stress.perturbations import jitter_timestamps


def test_timestamp_jitter_is_shared_within_each_sequence_frame() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqB", "seqB"],
            "frame_index": [0, 0, 1, 0, 0],
            "time_s": [10.0, 10.0, 20.0, 10.0, 10.0],
            "track_id": [1, 2, 1, 1, 2],
        }
    )

    perturbed = jitter_timestamps(
        radar,
        std_s=0.5,
        rng=np.random.default_rng(17),
    )
    offsets = perturbed["time_s"] - radar["time_s"]

    assert offsets.iloc[0] == offsets.iloc[1]
    assert offsets.iloc[3] == offsets.iloc[4]
    assert len(np.unique(offsets.to_numpy())) == 3


def test_timestamp_jitter_uses_time_for_rows_without_frame_index() -> None:
    frame = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "frame_index": [np.nan, np.nan, np.nan],
            "time_s": [10.0, 10.0, 20.0],
        }
    )

    perturbed = jitter_timestamps(
        frame,
        std_s=0.5,
        rng=np.random.default_rng(23),
    )
    offsets = perturbed["time_s"] - frame["time_s"]

    assert offsets.iloc[0] == offsets.iloc[1]
    assert offsets.iloc[0] != offsets.iloc[2]


def test_timestamp_jitter_preserves_rng_consumption_per_input_row() -> None:
    frame = pd.DataFrame(
        {
            "frame_index": [0, 0, 1],
            "time_s": [10.0, 10.0, 20.0],
        }
    )
    actual_rng = np.random.default_rng(31)
    jitter_timestamps(frame, std_s=0.5, rng=actual_rng)

    expected_rng = np.random.default_rng(31)
    expected_rng.normal(0.0, 0.5, len(frame))

    assert actual_rng.normal() == expected_rng.normal()
