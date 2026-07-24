from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
)


_MASKED_FIELDS = (
    "top_k",
    "iterations",
    "default_sigma_m",
    "sigma_min_m",
    "sigma_max_m",
    "score_weight",
    "temperature",
    "sigma_log_weight",
    "huber_delta",
    "smoothness_weight",
    "anchor_weight",
    "tolerance_m",
    "target_time_tolerance_s",
    "uniform_weight_floor",
    "branch_balance",
    "source_balance",
    "responsibility_floor",
    "min_measurement_precision",
    "max_measurement_precision",
)


@pytest.mark.parametrize("field", _MASKED_FIELDS)
def test_candidate_mixture_map_rejects_masked_config_scalars(field: str) -> None:
    defaults = CandidateMixtureMapConfig()
    masked = np.ma.array(getattr(defaults, field), mask=True)
    config = replace(defaults, **{field: masked})

    with pytest.raises(ValueError, match=field):
        run_candidate_mixture_map(pd.DataFrame(), config=config)


def test_candidate_mixture_map_rejects_masked_singleton() -> None:
    config = replace(CandidateMixtureMapConfig(), top_k=np.ma.masked)

    with pytest.raises(ValueError, match="top_k"):
        run_candidate_mixture_map(pd.DataFrame(), config=config)
