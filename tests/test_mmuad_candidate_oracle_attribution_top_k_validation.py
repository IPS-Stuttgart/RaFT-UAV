from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "lidar"],
            "track_id": ["best", "second"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [0.5, 1.0],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        0,
        -1,
        1.5,
        True,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.asarray([1]),
    ],
)
def test_candidate_oracle_attribution_rejects_malformed_top_k_values(
    invalid_top_k: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k_values must contain only positive integers",
    ):
        build_candidate_oracle_attribution_tables(
            _candidate_rows(),
            _truth_rows(),
            top_k_values=(invalid_top_k,),
        )


def test_candidate_oracle_attribution_rejects_string_top_k_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="top_k_values must contain only positive integers",
    ):
        build_candidate_oracle_attribution_tables(
            _candidate_rows(),
            _truth_rows(),
            top_k_values="12",
        )


def test_candidate_oracle_attribution_normalizes_integer_valued_top_k() -> None:
    frame_rows, pooled, _by_branch, _by_source = (
        build_candidate_oracle_attribution_tables(
            _candidate_rows(),
            _truth_rows(),
            top_k_values=(np.int64(2), np.asarray(1.0), 2.0),
        )
    )

    assert "oracle_top1_3d_m" in frame_rows.columns
    assert "oracle_top2_3d_m" in frame_rows.columns
    assert "oracle_top3_3d_m" not in frame_rows.columns
    assert "oracle_top1_3d_m_mse" in pooled.columns
    assert "oracle_top2_3d_m_mse" in pooled.columns
