from __future__ import annotations

import math
import runpy
import sys

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    select_spatial_hypothesis_group_topk,
)


@pytest.mark.parametrize(
    "field_name",
    ["diversity_weight", "diversity_scale_m", "diversity_cap_m"],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_spatial_group_topk_rejects_nonfinite_diversity_settings(
    field_name: str,
    value: float,
) -> None:
    config = SpatialHypothesisGroupTopKConfig(**{field_name: value})

    with pytest.raises(ValueError, match=rf"{field_name} must be finite"):
        select_spatial_hypothesis_group_topk(
            pd.DataFrame(),
            selection_config=config,
        )


def test_spatial_group_topk_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.candidate_mixture_group_spatial_topk"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
