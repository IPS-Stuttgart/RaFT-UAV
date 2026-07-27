from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.mot import _greedy_truth_matches, compute_multi_object_metrics


@pytest.mark.parametrize(
    "invalid_distance",
    [
        np.array(np.complex64(25.0 + 4.0j), dtype=object),
        np.array(np.complex128(25.0 + 4.0j), dtype=object),
        np.ma.array(np.complex64(25.0 + 4.0j), dtype=object, mask=False),
    ],
)
def test_multi_object_metrics_rejects_object_wrapped_complex_match_distance(
    invalid_distance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="match_distance_m must be finite and nonnegative",
    ):
        compute_multi_object_metrics(
            pd.DataFrame(),
            None,
            match_distance_m=invalid_distance,
        )


def test_greedy_truth_matches_rejects_object_wrapped_complex_distance() -> None:
    invalid_distance = np.array(np.complex64(25.0 + 4.0j), dtype=object)

    with pytest.raises(
        ValueError,
        match="match_distance_m must be finite and nonnegative",
    ):
        _greedy_truth_matches(
            pd.DataFrame(),
            pd.DataFrame(),
            max_distance_m=invalid_distance,
        )
