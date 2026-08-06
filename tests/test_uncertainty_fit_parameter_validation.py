from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [1.0, 2.0],
            "north_m": [1.0, 2.0],
        }
    )
    return truth, rf


def _fit(**kwargs):
    truth, rf = _frames()
    return fit_heteroscedastic_uncertainty_model(
        rf=rf,
        radar=None,
        truth=truth,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ridge_lambda": -1.0}, "ridge_lambda must be nonnegative"),
        ({"ridge_lambda": np.nan}, "ridge_lambda must be a finite number"),
        ({"ridge_lambda": True}, "ridge_lambda must be a finite number"),
        ({"max_time_delta_s": -1.0}, "max_time_delta_s must be nonnegative"),
        (
            {"max_time_delta_s": np.inf},
            "max_time_delta_s must be a finite number",
        ),
    ],
)
def test_fit_rejects_invalid_scalar_parameters(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _fit(**kwargs)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, True])
def test_fit_rejects_invalid_standard_deviation_bounds(value: object) -> None:
    with pytest.raises(ValueError, match="min_std_m.*must be"):
        _fit(min_std_m={"rf": {"east": value}})

    with pytest.raises(ValueError, match="max_std_m.*must be"):
        _fit(max_std_m={"rf": {"east": value}})


def test_fit_rejects_inverted_standard_deviation_bounds() -> None:
    with pytest.raises(ValueError, match="min_std_m must not exceed max_std_m"):
        _fit(
            min_std_m={"rf": {"east": 100.0}},
            max_std_m={"rf": {"east": 10.0}},
        )


def test_fit_rejects_unknown_bound_source_and_dimension() -> None:
    with pytest.raises(ValueError, match="unknown source 'lidar'"):
        _fit(min_std_m={"lidar": {"east": 1.0}})

    with pytest.raises(ValueError, match="unknown dimension 'eest'"):
        _fit(max_std_m={"rf": {"eest": 100.0}})


def test_fit_accepts_zero_penalty_exact_alignment_and_equal_bounds() -> None:
    model = _fit(
        ridge_lambda=0.0,
        max_time_delta_s=0.0,
        min_std_m={"rf": {"east": 10.0}},
        max_std_m={"rf": {"east": 10.0}},
    )

    east_head = next(
        head for head in model.heads if head.source == "rf" and head.dimension == "east"
    )
    assert east_head.min_std_m == 10.0
    assert east_head.max_std_m == 10.0
    assert model.metadata["ridge_lambda"] == 0.0
    assert model.metadata["max_time_delta_s"] == 0.0


def test_fit_controls_override_colliding_caller_metadata() -> None:
    metadata = {
        "ridge_lambda": 999.0,
        "max_time_delta_s": 999.0,
        "experiment": "synthetic-regression",
    }

    model = _fit(
        ridge_lambda=np.float64(0.25),
        max_time_delta_s=np.float64(0.5),
        metadata=metadata,
    )

    assert model.metadata["ridge_lambda"] == 0.25
    assert model.metadata["max_time_delta_s"] == 0.5
    assert model.metadata["experiment"] == "synthetic-regression"
    assert metadata["ridge_lambda"] == 999.0
    assert metadata["max_time_delta_s"] == 999.0
