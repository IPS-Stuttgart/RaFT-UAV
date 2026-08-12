import numpy as np
import pytest

from raft_uav.evaluation.metrics import interpolate_positions_at_times


def test_interpolate_positions_at_times_rejects_zero_coordinate_dimensions() -> None:
    with pytest.raises(ValueError, match="at least one coordinate dimension"):
        interpolate_positions_at_times(
            np.array([0.0]),
            np.empty((1, 0)),
            np.array([0.0]),
        )
