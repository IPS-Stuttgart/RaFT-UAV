from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.candidate_pool_compare as pool_compare


_INVALID_TOP_K_VALUES = (
    True,
    np.bool_(False),
    1.5,
    np.float64(2.5),
    np.array([3]),
    np.array(True),
    np.ma.masked,
    1 + 0j,
)


@pytest.mark.parametrize("value", _INVALID_TOP_K_VALUES)
def test_candidate_pool_compare_rejects_lossy_top_k_values(value: object) -> None:
    with pytest.raises(ValueError, match="top_k_values"):
        pool_compare.build_candidate_pool_compare_tables(
            pd.DataFrame(),
            {},
            pd.DataFrame(),
            top_k_values=[value],
        )


def test_candidate_pool_compare_rejects_cyclic_top_k_array() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(ValueError, match="top_k_values"):
        pool_compare.build_candidate_pool_compare_tables(
            pd.DataFrame(),
            {},
            pd.DataFrame(),
            top_k_values=[cyclic],
        )


def test_candidate_pool_compare_preserves_integer_like_top_k_values() -> None:
    normalized = pool_compare._normalize_top_k_values(
        [3, "1", np.int64(2), 0, -4, np.array(3)],
    )

    assert normalized == (1, 2, 3)
