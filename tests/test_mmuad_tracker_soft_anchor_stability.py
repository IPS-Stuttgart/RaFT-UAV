from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker


def _large_residual_candidates() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["sequence-1", "sequence-1", "sequence-1"],
                "time_s": [0.0, 1.0, 1.0],
                "source": ["radar", "radar", "radar"],
                "track_id": ["anchor", "clutter", "anchor"],
                "x_m": [0.0, 6.0e307, 0.0],
                "y_m": [0.0, 8.0e307, 0.0],
                "z_m": [0.0, 0.0, 0.0],
            }
        )
    )


def test_soft_anchor_gate_handles_large_representable_norm() -> None:
    config = replace(
        TrackerConfig(),
        soft_anchor_gate_m=9.0e307,
    )

    with np.errstate(over="raise", invalid="raise"):
        output = run_mmuad_tracker(_large_residual_candidates(), config=config)

    clutter = output.estimates.loc[output.estimates["track_id"] == "clutter"].iloc[0]
    assert clutter["update_action"] == "soft_anchor_gated"
    np.testing.assert_allclose(
        clutter[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float),
        np.zeros(3),
    )


def test_soft_anchor_cap_preserves_direction_for_large_representable_norm() -> None:
    config = replace(
        TrackerConfig(),
        soft_anchor_gate_m=0.0,
        soft_anchor_cap_m=2.0,
    )

    with np.errstate(over="raise", invalid="raise"):
        output = run_mmuad_tracker(_large_residual_candidates(), config=config)

    clutter = output.estimates.loc[output.estimates["track_id"] == "clutter"].iloc[0]
    state_xy = clutter[["state_x_m", "state_y_m"]].to_numpy(dtype=float)
    assert clutter["update_action"] == "soft_anchor"
    assert np.isfinite(state_xy).all()
    assert np.hypot.reduce(np.abs(state_xy)) > 0.0
    np.testing.assert_allclose(
        state_xy / np.hypot.reduce(np.abs(state_xy)),
        np.array([0.6, 0.8]),
        rtol=1.0e-12,
        atol=0.0,
    )
