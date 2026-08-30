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


def _large_uncertainty_candidates(*, std_xy_m: float, std_z_m: float) -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["sequence-1", "sequence-1"],
                "time_s": [0.0, 1.0],
                "source": ["radar", "radar"],
                "track_id": ["anchor", "anchor"],
                "x_m": [0.0, 0.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 10.0],
                "std_xy_m": [std_xy_m, std_xy_m],
                "std_z_m": [std_z_m, std_z_m],
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


def test_selected_update_handles_finite_standard_deviation_square_overflow() -> None:
    candidates = _large_uncertainty_candidates(std_xy_m=1.0e308, std_z_m=1.0e308)

    with np.errstate(all="raise"):
        output = run_mmuad_tracker(candidates)

    states = output.estimates[
        ["state_x_m", "state_y_m", "state_z_m", "v_x_mps", "v_y_mps", "v_z_mps"]
    ].to_numpy(dtype=float)
    assert np.isfinite(states).all()


def test_anisotropic_measurement_covariance_keeps_precise_axis_update() -> None:
    candidates = _large_uncertainty_candidates(std_xy_m=1.0e100, std_z_m=1.0)

    with np.errstate(all="raise"):
        output = run_mmuad_tracker(candidates)

    final = output.estimates.sort_values("time_s").iloc[-1]
    assert final["update_action"] == "selected_update"
    assert float(final["state_z_m"]) > 1.0
    assert np.isfinite(
        final[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(dtype=float)
    ).all()
