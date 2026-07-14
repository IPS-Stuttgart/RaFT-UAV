import numpy as np
import pytest

from raft_uav.evaluation.metrics import nearest_time_indices


@pytest.mark.parametrize(
    "query_times",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
        np.array([0.0, np.nan]),
    ],
    ids=["nan", "positive-infinity", "negative-infinity", "mixed"],
)
def test_nearest_time_indices_rejects_nonfinite_query_times(query_times):
    with pytest.raises(
        ValueError,
        match="query_times_s must contain only finite timestamps",
    ):
        nearest_time_indices(np.array([0.0, 1.0, 2.0]), query_times)
