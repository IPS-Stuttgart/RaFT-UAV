from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_soft_class_ensemble import (
    _normalized_weight_map,
    _select_trim_fraction,
    build_soft_class_conditioned_estimate_ensemble,
)


def _inputs(path: Path = Path("missing.csv")) -> tuple[EstimateInput, ...]:
    return (EstimateInput("first", path),)


@pytest.mark.parametrize(
    "weight",
    [
        True,
        False,
        np.bool_(True),
        np.array(True),
        np.array([1.0]),
        np.nan,
        np.inf,
        1 + 0j,
        np.ma.masked,
    ],
)
def test_soft_class_ensemble_rejects_malformed_weights(weight: object) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _normalized_weight_map({"first": weight}, _inputs())


@pytest.mark.parametrize(
    "trim",
    [
        True,
        False,
        np.bool_(False),
        np.array(False),
        np.array([0.2]),
        np.nan,
        np.inf,
        0.2 + 0j,
        np.ma.masked,
    ],
)
def test_soft_class_ensemble_rejects_malformed_trim_fraction(trim: object) -> None:
    with pytest.raises(ValueError, match="finite real scalar"):
        _select_trim_fraction(trim, {})


def test_soft_class_ensemble_accepts_zero_dimensional_real_scalars() -> None:
    assert _normalized_weight_map(
        {"first": np.array(2.0)},
        _inputs(),
    ) == {"first": 1.0}
    assert _select_trim_fraction(np.array(0.25), {}) == pytest.approx(0.25)


def test_soft_class_ensemble_validates_class_weights_before_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_soft_class_conditioned_estimate_ensemble(
            _inputs(tmp_path / "missing.csv"),
            template=pd.DataFrame(),
            class_probabilities=pd.DataFrame(),
            weight_config={
                "global_weights": {"first": 1.0},
                "class_weights": {"0": {"first": True}},
            },
        )


def test_soft_class_ensemble_validates_config_trim_before_file_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="finite real scalar"):
        build_soft_class_conditioned_estimate_ensemble(
            _inputs(tmp_path / "missing.csv"),
            template=pd.DataFrame(),
            class_probabilities=pd.DataFrame(),
            weight_config={
                "global_weights": {"first": 1.0},
                "trim_fraction": False,
            },
        )
