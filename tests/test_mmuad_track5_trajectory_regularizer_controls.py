from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_trajectory_regularizer import regularize_track5_estimates
from raft_uav.mmuad.track5_trajectory_regularizer import run_track5_trajectory_regularizer


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
        }
    )


def _nested_scalar(value: object) -> np.ndarray:
    outer = np.empty((), dtype=object)
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "smoothness_weight",
            np.nan,
            "smoothness_weight must be a finite non-negative real scalar",
        ),
        (
            "smoothness_weight",
            np.inf,
            "smoothness_weight must be a finite non-negative real scalar",
        ),
        (
            "smoothness_weight",
            True,
            "smoothness_weight must be a finite non-negative real scalar",
        ),
        (
            "smoothness_weight",
            -1.0,
            "smoothness_weight must be a finite non-negative real scalar",
        ),
        (
            "huber_delta_m",
            np.nan,
            "huber_delta_m must be a finite positive real scalar",
        ),
        (
            "huber_delta_m",
            0.0,
            "huber_delta_m must be a finite positive real scalar",
        ),
        (
            "huber_delta_m",
            False,
            "huber_delta_m must be a finite positive real scalar",
        ),
        (
            "observation_sigma_m",
            np.inf,
            "observation_sigma_m must be a finite positive real scalar",
        ),
        (
            "observation_sigma_m",
            0.0,
            "observation_sigma_m must be a finite positive real scalar",
        ),
        (
            "observation_sigma_m",
            np.array([1.0]),
            "observation_sigma_m must be a finite positive real scalar",
        ),
        (
            "observation_sigma_m",
            _nested_scalar(True),
            "observation_sigma_m must be a finite positive real scalar",
        ),
        ("iterations", True, "iterations must be a positive finite integer"),
        ("iterations", 0, "iterations must be a positive finite integer"),
        ("iterations", 1.5, "iterations must be a positive finite integer"),
        ("iterations", np.inf, "iterations must be a positive finite integer"),
    ],
)
def test_regularizer_rejects_malformed_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "smoothness_weight": 10.0,
        "huber_delta_m": 25.0,
        "iterations": 5,
        "observation_sigma_m": 10.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        regularize_track5_estimates(_estimates(), **kwargs)


def test_regularizer_accepts_lossless_serialized_scalars() -> None:
    regularized, diagnostics = regularize_track5_estimates(
        _estimates(),
        smoothness_weight="0.0",
        huber_delta_m=np.float64(25.0),
        iterations="2.0",
        observation_sigma_m=_nested_scalar(10.0),
    )

    assert len(regularized) == 3
    assert diagnostics.loc[0, "smoothness_weight"] == pytest.approx(0.0)
    assert diagnostics.loc[0, "huber_delta_m"] == pytest.approx(25.0)
    assert diagnostics.loc[0, "iterations"] == 2
    assert diagnostics.loc[0, "observation_sigma_m"] == pytest.approx(10.0)


def test_run_rejects_invalid_controls_before_output_creation(tmp_path: Path) -> None:
    output_dir = tmp_path / "regularized"

    with pytest.raises(
        ValueError,
        match="smoothness_weight must be a finite non-negative real scalar",
    ):
        run_track5_trajectory_regularizer(
            estimates=pd.DataFrame(),
            template=pd.DataFrame(),
            output_dir=output_dir,
            smoothness_weight=np.nan,
        )

    assert not output_dir.exists()
