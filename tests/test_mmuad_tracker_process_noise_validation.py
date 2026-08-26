from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker


def _candidate_frame() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["sequence-1", "sequence-1"],
                "time_s": [0.0, 1.0],
                "source": ["radar", "radar"],
                "x_m": [0.0, 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
            }
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_tracker_rejects_invalid_process_noise(value: object) -> None:
    config = replace(TrackerConfig(), acceleration_std_mps2=value)

    with pytest.raises(
        ValueError,
        match=r"acceleration_std_mps2 must be a finite non-negative real scalar",
    ):
        run_mmuad_tracker(_candidate_frame(), config=config)


def test_tracker_accepts_and_normalizes_non_negative_process_noise() -> None:
    output = run_mmuad_tracker(
        _candidate_frame(),
        config=replace(TrackerConfig(), acceleration_std_mps2="0"),
    )

    assert not output.estimates.empty
    assert np.isfinite(
        output.estimates[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all()
