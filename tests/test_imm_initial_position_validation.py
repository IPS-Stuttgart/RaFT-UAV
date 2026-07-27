from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker


@pytest.mark.parametrize(
    "initial_position",
    [
        np.array([0.0, np.nan, 0.0]),
        np.array([0.0 + 1.0j, 0.0, 0.0]),
        np.ma.array([0.0, 1.0, 2.0], mask=[False, True, False]),
    ],
)
def test_imm_tracker_rejects_invalid_initial_positions(initial_position) -> None:
    with pytest.raises(ValueError, match="initial_position"):
        AsyncInteractingMultipleModelTracker(
            initial_position=initial_position,
            initial_time_s=0.0,
        )


def test_imm_tracker_accepts_unmasked_initial_state() -> None:
    initial_state = np.ma.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        mask=False,
    )

    tracker = AsyncInteractingMultipleModelTracker(
        initial_position=initial_state,
        initial_time_s=0.0,
    )

    np.testing.assert_allclose(tracker.state, initial_state.data)
