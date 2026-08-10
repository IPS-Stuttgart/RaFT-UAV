from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    compute_candidate_responsibilities,
    run_candidate_mixture_map,
)


_INVALID_CONFIGS = (False, 0, 0.0, "", (), [], {})


@pytest.mark.parametrize("config", _INVALID_CONFIGS)
def test_candidate_mixture_map_rejects_falsy_non_config(config: object) -> None:
    with pytest.raises(TypeError, match="CandidateMixtureMapConfig"):
        run_candidate_mixture_map(pd.DataFrame(), config=config)


@pytest.mark.parametrize("config", _INVALID_CONFIGS)
def test_candidate_responsibilities_rejects_falsy_non_config(config: object) -> None:
    with pytest.raises(TypeError, match="CandidateMixtureMapConfig"):
        compute_candidate_responsibilities(
            pd.DataFrame(),
            np.zeros(3),
            config=config,
        )
