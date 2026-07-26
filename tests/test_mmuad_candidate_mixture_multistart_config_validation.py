from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map_multistart as multistart


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        -1,
        1.5,
        np.nan,
        np.inf,
        np.array([1]),
        1 + 0j,
    ],
)
def test_multistart_rejects_invalid_max_branch_starts(value) -> None:
    config = multistart.CandidateMixtureMultiStartConfig(max_branch_starts=value)

    with pytest.raises(ValueError, match="max_branch_starts"):
        multistart.build_candidate_mixture_initializations(
            pd.DataFrame(),
            multistart_config=config,
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        -0.1,
        1.1,
        np.nan,
        np.inf,
        np.array([0.5]),
        0.5 + 0j,
    ],
)
def test_multistart_rejects_invalid_branch_frame_fraction(value) -> None:
    config = multistart.CandidateMixtureMultiStartConfig(
        min_branch_frame_fraction=value,
    )

    with pytest.raises(ValueError, match="min_branch_frame_fraction"):
        multistart.build_candidate_mixture_initializations(
            pd.DataFrame(),
            multistart_config=config,
        )


@pytest.mark.parametrize(
    ("max_branch_starts", "min_branch_frame_fraction"),
    [
        (0, 0.0),
        (2.0, np.float64(0.25)),
        (np.array(3), np.array(0.5)),
        ("4", "0.75"),
    ],
)
def test_multistart_accepts_exact_scalar_controls(
    max_branch_starts,
    min_branch_frame_fraction,
) -> None:
    config = multistart.CandidateMixtureMultiStartConfig(
        max_branch_starts=max_branch_starts,
        min_branch_frame_fraction=min_branch_frame_fraction,
    )

    starts = multistart.build_candidate_mixture_initializations(
        pd.DataFrame(),
        multistart_config=config,
    )

    assert starts == {"core-default": None}
