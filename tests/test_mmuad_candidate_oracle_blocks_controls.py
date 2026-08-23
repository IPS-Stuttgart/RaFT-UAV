from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_blocks import (
    build_candidate_oracle_block_tables,
)


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 10.0],
            "oracle_all_3d_m": [1.0, 1.0],
            "oracle_all_rank": [1, 1],
        }
    )


@pytest.mark.parametrize(
    "top_k",
    [
        True,
        0,
        -1,
        1.5,
        float("nan"),
        float("inf"),
        [1],
        np.array([1]),
    ],
)
def test_candidate_oracle_blocks_reject_invalid_top_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        build_candidate_oracle_block_tables(_frame_rows(), top_k=top_k)


@pytest.mark.parametrize(
    "threshold",
    [
        True,
        -1.0,
        float("nan"),
        float("inf"),
        [5.0],
        np.array([5.0]),
    ],
)
def test_candidate_oracle_blocks_reject_invalid_threshold(
    threshold: object,
) -> None:
    with pytest.raises(ValueError, match="oracle_error_threshold_m"):
        build_candidate_oracle_block_tables(
            _frame_rows(),
            oracle_error_threshold_m=threshold,
        )


@pytest.mark.parametrize(
    "max_gap_s",
    [
        True,
        -1.0,
        float("nan"),
        float("inf"),
        [1.0],
        np.array([1.0]),
    ],
)
def test_candidate_oracle_blocks_reject_invalid_max_gap(
    max_gap_s: object,
) -> None:
    with pytest.raises(ValueError, match="max_gap_s"):
        build_candidate_oracle_block_tables(_frame_rows(), max_gap_s=max_gap_s)


def test_candidate_oracle_blocks_preserve_integral_serialized_controls() -> None:
    blocks, _ = build_candidate_oracle_block_tables(
        _frame_rows(),
        oracle_error_threshold_m="5.0",
        top_k="2.0",
        max_gap_s="1.0",
    )

    assert blocks["frame_count"].tolist() == [1, 1]


def test_candidate_oracle_blocks_accept_numpy_scalar_controls() -> None:
    blocks, _ = build_candidate_oracle_block_tables(
        _frame_rows(),
        oracle_error_threshold_m=np.float64(5.0),
        top_k=np.int64(2),
        max_gap_s=np.float64(1.0),
    )

    assert blocks["frame_count"].tolist() == [1, 1]
