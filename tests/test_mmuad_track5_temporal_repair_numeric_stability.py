from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_temporal_repair import _sequence_diagnostics
from raft_uav.mmuad.track5_temporal_repair import repair_track5_temporal_spikes


def _normalized_submission(
    x_m: list[float],
    y_m: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(x_m),
            "time_s": np.arange(len(x_m), dtype=float),
            "state_x_m": x_m,
            "state_y_m": y_m,
            "state_z_m": [0.0] * len(x_m),
            "Classification": [2] * len(x_m),
        }
    )


def test_temporal_repair_keeps_representable_large_diagonal_norms() -> None:
    submission = _normalized_submission(
        [0.0, 6.0e307, 0.0, 0.0],
        [0.0, 8.0e307, 0.0, 0.0],
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        initial = _sequence_diagnostics(submission, iteration=1)
        repaired, diagnostics = repair_track5_temporal_spikes(
            submission,
            max_speed_mps=9.0e307,
            max_interpolation_residual_m=1.0e307,
            iterations=1,
        )

    spike = initial.iloc[1]
    assert spike["incoming_speed_mps"] == pytest.approx(1.0e308)
    assert spike["outgoing_speed_mps"] == pytest.approx(1.0e308)
    assert spike["neighbor_direct_speed_mps"] == 0.0
    assert spike["interpolation_residual_m"] == pytest.approx(1.0e308)

    repaired_spike = repaired.iloc[1]
    assert repaired_spike["state_x_m"] == 0.0
    assert repaired_spike["state_y_m"] == 0.0
    assert repaired_spike["Classification"] == 2
    changed = diagnostics.loc[diagnostics["repaired"]]
    assert len(changed) == 1
    assert changed.iloc[0]["repair_displacement_m"] == pytest.approx(1.0e308)


def test_temporal_repair_uses_stable_convex_interpolation_for_extremes() -> None:
    submission = _normalized_submission(
        [-1.0e308, 0.0, 1.0e308, 1.0e308],
        [0.0, 1.0e307, 0.0, 0.0],
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        initial = _sequence_diagnostics(submission, iteration=1)
        repaired, diagnostics = repair_track5_temporal_spikes(
            submission,
            max_speed_mps=1.002e308,
            max_interpolation_residual_m=1.0e306,
            iterations=1,
        )

    spike = initial.iloc[1]
    numeric_columns = [
        "incoming_speed_mps",
        "outgoing_speed_mps",
        "neighbor_direct_speed_mps",
        "interpolation_residual_m",
        "interp_x_m",
        "interp_y_m",
        "interp_z_m",
    ]
    assert np.isfinite(spike[numeric_columns].to_numpy(float)).all()
    assert spike["interp_x_m"] == 0.0
    assert spike["interp_y_m"] == 0.0
    assert spike["incoming_speed_mps"] == pytest.approx(np.hypot(1.0e308, 1.0e307))
    assert spike["outgoing_speed_mps"] == pytest.approx(np.hypot(1.0e308, 1.0e307))
    assert spike["neighbor_direct_speed_mps"] == pytest.approx(1.0e308)
    assert spike["interpolation_residual_m"] == pytest.approx(1.0e307)

    repaired_spike = repaired.iloc[1]
    assert repaired_spike["state_x_m"] == 0.0
    assert repaired_spike["state_y_m"] == 0.0
    changed = diagnostics.loc[diagnostics["repaired"]]
    assert len(changed) == 1
    assert changed.iloc[0]["repair_displacement_m"] == pytest.approx(1.0e307)
