from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_blocks import (
    _to_bool_series,
    build_candidate_oracle_block_tables,
)


def test_candidate_oracle_blocks_parse_serialized_topk_flags() -> None:
    values = pd.Series(
        [False, True, "False", " TRUE ", "0", "1", "no", "yes", None],
        index=[10, 11, 12, 13, 14, 15, 16, 17, 18],
    )

    parsed = _to_bool_series(values)

    assert parsed.tolist() == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]
    assert parsed.index.tolist() == values.index.tolist()


@pytest.mark.parametrize("value", [2, -1, 0.5, np.inf, "maybe"])
def test_candidate_oracle_blocks_reject_ambiguous_topk_flags(value: object) -> None:
    values = pd.Series([value], index=[42])

    with pytest.raises(
        ValueError,
        match=r"invalid Boolean values at rows \[42\]",
    ):
        _to_bool_series(values)


def test_block_builder_rejects_corrupted_persisted_topk_flags() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "oracle_all_3d_m": [1.0],
            "oracle_all_rank": [1.0],
            "oracle_in_top5": [2],
        },
        index=[7],
    )

    with pytest.raises(
        ValueError,
        match=r"invalid Boolean values at rows \[7\]",
    ):
        build_candidate_oracle_block_tables(frame_rows, top_k=5)
