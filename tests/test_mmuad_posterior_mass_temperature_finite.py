from __future__ import annotations

import math
import runpy
import sys

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
)


@pytest.mark.parametrize("temperature", [math.nan, math.inf, -math.inf])
def test_posterior_mass_group_topk_rejects_nonfinite_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValueError, match="posterior_temperature must be finite"):
        select_posterior_mass_hypothesis_group_topk(
            pd.DataFrame(),
            selection_config=PosteriorMassGroupTopKConfig(
                posterior_temperature=temperature,
            ),
        )


def test_posterior_mass_group_topk_package_supports_python_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "raft_uav.mmuad.candidate_mixture_group_mass_topk"
    monkeypatch.setattr(sys, "argv", [module_name, "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module(module_name, run_name="__main__")

    assert exc_info.value.code == 0
