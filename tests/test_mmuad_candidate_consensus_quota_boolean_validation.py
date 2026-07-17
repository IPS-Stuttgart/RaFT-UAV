from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_consensus_quota import (
    build_consensus_quota_reservoir,
)


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "true",
        0,
        1,
        None,
        np.nan,
        np.array([True]),
    ],
)
def test_consensus_quota_rejects_truthiness_coercion_for_boolean_control(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="exclude_same_origin_support must be a Boolean scalar",
    ):
        build_consensus_quota_reservoir(
            pd.DataFrame(),
            exclude_same_origin_support=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        np.array(False),
    ],
)
def test_consensus_quota_accepts_boolean_scalars(value: object) -> None:
    reservoir = build_consensus_quota_reservoir(
        pd.DataFrame(),
        exclude_same_origin_support=value,
    )

    assert reservoir.rows.empty
