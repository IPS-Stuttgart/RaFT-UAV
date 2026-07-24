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


def test_nearest_time_indices_rejects_masked_query_timestamps():
    with pytest.raises(
        ValueError,
        match="query_times_s must contain only finite timestamps",
    ):
        nearest_time_indices(
            np.array([0.0, 1.0]),
            np.ma.array([0.75], mask=[True]),
        )


def test_nearest_time_indices_ignores_masked_reference_timestamps():
    indices = nearest_time_indices(
        np.ma.array([0.0, 100.0], mask=[False, True]),
        np.array([90.0]),
    )

    np.testing.assert_array_equal(indices, np.array([0]))
