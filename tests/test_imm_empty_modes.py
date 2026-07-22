import numpy as np
import pytest

from raft_uav.baselines.imm import AsyncInteractingMultipleModelTracker


def test_imm_tracker_rejects_explicit_empty_mode_bank() -> None:
    with pytest.raises(ValueError, match="modes must contain at least one IMMMode"):
        AsyncInteractingMultipleModelTracker(
            initial_position=np.zeros(3),
            initial_time_s=0.0,
            modes=[],
        )
