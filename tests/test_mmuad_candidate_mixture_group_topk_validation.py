from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_topk import (
    HypothesisGroupTopKConfig,
    _validate_selection_config,
    select_hypothesis_group_topk,
)


@pytest.mark.parametrize("field", ["group_top_k", "max_siblings_per_group"])
@pytest.mark.parametrize("value", [1.5, True, np.nan, np.array([1])])
def test_group_topk_rejects_malformed_integer_controls(
    field: str,
    value: object,
) -> None:
    config = HypothesisGroupTopKConfig(**{field: value})

    with pytest.raises(
        ValueError,
        match=rf"{field} must be a non-negative integer",
    ):
        select_hypothesis_group_topk(
            pd.DataFrame(),
            selection_config=config,
        )


def test_group_topk_accepts_integer_valued_scalars() -> None:
    _validate_selection_config(
        HypothesisGroupTopKConfig(
            group_top_k=np.int64(2),
            max_siblings_per_group=2.0,
        )
    )
