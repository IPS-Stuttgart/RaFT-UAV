import numpy as np
import pytest

from raft_uav.evaluation.metrics import nearest_time_indices


@pytest.mark.parametrize("query_time", [np.nan, np.inf, -np.inf])
def test_nearest_time_indices_rejects_nonfinite_query_timestamps(query_time):
    with pytest.raises(
        ValueError,
        match="query_times_s must contain only finite timestamps",
    ):
        nearest_time_indices(
            np.array([0.0, 1.0]),
            np.array([query_time]),
        )
