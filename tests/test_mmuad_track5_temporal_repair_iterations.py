from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_temporal_repair import repair_track5_temporal_spikes


def _normalized_submission_with_spike() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "state_x_m": [0.0, 100.0, 2.0, 3.0],
            "state_y_m": [0.0, 0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0, 0.0],
            "Classification": [2, 2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "iterations",
    [0, -1, 1.5, True, np.bool_(False), np.array([2]), np.ma.masked],
)
def test_temporal_repair_rejects_invalid_iteration_controls(iterations: object) -> None:
    submission = _normalized_submission_with_spike()
    original = submission.copy(deep=True)

    with pytest.raises(ValueError, match="iterations must be an exact positive integer"):
        repair_track5_temporal_spikes(
            submission,
            max_speed_mps=20.0,
            max_interpolation_residual_m=10.0,
            iterations=iterations,
        )

    pd.testing.assert_frame_equal(submission, original)


def test_temporal_repair_accepts_exact_integer_equivalent_iterations() -> None:
    repaired, diagnostics = repair_track5_temporal_spikes(
        _normalized_submission_with_spike(),
        max_speed_mps=20.0,
        max_interpolation_residual_m=10.0,
        iterations=np.array(2.0),
    )

    assert repaired.loc[repaired["time_s"] == 1.0, "state_x_m"].item() == pytest.approx(1.0)
    assert diagnostics["repaired"].sum() == 1
