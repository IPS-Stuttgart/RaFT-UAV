import numpy as np
import pandas as pd

from raft_uav.research.factor_graph import _initial_positions


def test_initial_positions_sorts_unsorted_initial_times_before_interpolation() -> None:
    times = np.array([0.0, 1.0, 2.0])
    initial = pd.DataFrame(
        {
            "time_s": [2.0, 0.0, 1.0],
            "east_m": [20.0, 0.0, 10.0],
            "north_m": [2.0, 0.0, 1.0],
            "up_m": [-2.0, 0.0, -1.0],
        }
    )

    positions = _initial_positions(times, pd.DataFrame(), initial)

    np.testing.assert_allclose(
        positions,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 1.0, -1.0],
                [20.0, 2.0, -2.0],
            ]
        ),
    )
