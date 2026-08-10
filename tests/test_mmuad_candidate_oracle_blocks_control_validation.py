from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_blocks import build_candidate_oracle_block_tables


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-a"],
            "time_s": [0.0, 0.5],
            "oracle_all_3d_m": [0.5, 0.6],
            "oracle_all_rank": [1, 2],
            "oracle_in_top3": [True, True],
        }
    )


@pytest.mark.parametrize("value", [True, False, 1.5, 0, -1, np.array([3])])
def test_candidate_oracle_blocks_reject_invalid_top_k(value: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer scalar"):
        build_candidate_oracle_block_tables(_frame_rows(), top_k=value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("oracle_error_threshold_m", True),
        ("oracle_error_threshold_m", float("nan")),
        ("oracle_error_threshold_m", float("inf")),
        ("oracle_error_threshold_m", -0.1),
        ("max_gap_s", False),
        ("max_gap_s", float("nan")),
        ("max_gap_s", float("-inf")),
        ("max_gap_s", -0.1),
    ],
)
def test_candidate_oracle_blocks_reject_invalid_float_controls(
    name: str,
    value: object,
) -> None:
    kwargs = {name: value}
    with pytest.raises(
        ValueError,
        match=rf"{name} must be a finite non-negative real scalar",
    ):
        build_candidate_oracle_block_tables(_frame_rows(), **kwargs)


def test_candidate_oracle_blocks_accept_zero_dimensional_real_controls() -> None:
    blocks, _ = build_candidate_oracle_block_tables(
        _frame_rows(),
        oracle_error_threshold_m=np.array(5.0),
        top_k=np.array(3),
        max_gap_s=np.array(1.0),
    )

    assert int(blocks["frame_count"].sum()) == 2


def test_candidate_oracle_blocks_validate_controls_before_empty_fast_path() -> None:
    with pytest.raises(ValueError, match="top_k must be a positive integer scalar"):
        build_candidate_oracle_block_tables(pd.DataFrame(), top_k=True)
