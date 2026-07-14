from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.track5_class_conditioned_ensemble import _normalized_weight_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def _inputs() -> tuple[EstimateInput, ...]:
    return (
        EstimateInput("a", Path("a.csv")),
        EstimateInput("b", Path("b.csv")),
    )


@pytest.mark.parametrize("bad_weight", [np.nan, np.inf, -np.inf, -1.0])
def test_class_conditioned_weight_map_rejects_invalid_values(
    bad_weight: float,
) -> None:
    with pytest.raises(ValueError, match="finite and non-negative for a"):
        _normalized_weight_map({"a": bad_weight, "b": 1.0}, _inputs())


def test_class_conditioned_weight_map_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="weight map must be an object"):
        _normalized_weight_map([1.0, 0.0], _inputs())


def test_class_conditioned_weight_map_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown estimate labels.*typo"):
        _normalized_weight_map({"a": 1.0, "typo": 1.0}, _inputs())


def test_class_conditioned_weight_map_normalizes_extreme_finite_values() -> None:
    maximum = np.finfo(np.float64).max

    weights = _normalized_weight_map({"a": maximum, "b": maximum}, _inputs())

    assert weights == pytest.approx({"a": 0.5, "b": 0.5})
