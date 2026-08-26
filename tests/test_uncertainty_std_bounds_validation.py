"""Regression tests for uncertainty standard-deviation bound validation."""

from __future__ import annotations

import pytest

from raft_uav.uncertainty import VarianceHead


def _head_kwargs(*, min_std_m: float, max_std_m: float) -> dict[str, object]:
    return {
        "source": "rf",
        "dimension": "east",
        "feature_names": ("intercept",),
        "coefficients": (0.0,),
        "min_std_m": min_std_m,
        "max_std_m": max_std_m,
        "training_rows": 1,
    }


@pytest.mark.parametrize(
    ("min_std_m", "max_std_m", "message"),
    [
        (0.0, 10.0, "min_std_m must be positive"),
        (-1.0, 10.0, "min_std_m must be positive"),
        (1.0, 0.0, "max_std_m must be positive"),
        (1.0, -10.0, "max_std_m must be positive"),
        (11.0, 10.0, "min_std_m must not exceed max_std_m"),
    ],
)
def test_variance_head_rejects_invalid_std_bounds_on_construction(
    min_std_m: float,
    max_std_m: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VarianceHead(**_head_kwargs(min_std_m=min_std_m, max_std_m=max_std_m))


@pytest.mark.parametrize(
    ("min_std_m", "max_std_m", "message"),
    [
        (0.0, 10.0, "min_std_m must be positive"),
        (-1.0, 10.0, "min_std_m must be positive"),
        (1.0, 0.0, "max_std_m must be positive"),
        (1.0, -10.0, "max_std_m must be positive"),
        (11.0, 10.0, "min_std_m must not exceed max_std_m"),
    ],
)
def test_variance_head_rejects_invalid_std_bounds_from_dict(
    min_std_m: float,
    max_std_m: float,
    message: str,
) -> None:
    payload = _head_kwargs(min_std_m=min_std_m, max_std_m=max_std_m)
    payload["feature_names"] = ["intercept"]
    payload["coefficients"] = [0.0]
    with pytest.raises(ValueError, match=message):
        VarianceHead.from_dict(payload)
