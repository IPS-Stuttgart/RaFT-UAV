from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.bias import (
    RF_TARGET_COLUMNS,
    bias_training_rows,
    fit_bias_correction_models,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)


_INVALID_NONNEGATIVE_REALS = [
    np.nan,
    np.inf,
    -np.inf,
    True,
    np.array(True),
    np.array([1.0]),
]


@pytest.mark.parametrize("time_gate_s", _INVALID_NONNEGATIVE_REALS)
def test_bias_example_builder_rejects_invalid_time_gate_before_empty_fast_path(
    time_gate_s: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="time_gate_s must be a finite non-negative real scalar",
    ):
        make_bias_training_examples(
            pd.DataFrame(),
            pd.DataFrame(),
            source="rf",
            target_columns=RF_TARGET_COLUMNS,
            time_gate_s=time_gate_s,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ridge_alpha", np.nan, "ridge_alpha must be a finite non-negative real scalar"),
        ("ridge_alpha", np.inf, "ridge_alpha must be a finite non-negative real scalar"),
        ("ridge_alpha", True, "ridge_alpha must be a finite non-negative real scalar"),
        ("min_samples", np.nan, "min_samples must be a positive integer"),
        ("min_samples", True, "min_samples must be a positive integer"),
        ("min_samples", np.array(True), "min_samples must be a positive integer"),
    ],
)
def test_bias_fit_rejects_invalid_controls_before_example_schema_access(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "source": "rf",
        "target_columns": RF_TARGET_COLUMNS,
        "time_gate_s": 2.0,
        "ridge_alpha": 0.0,
        "min_samples": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        fit_sensor_bias_correction_from_examples(
            pd.DataFrame(),
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_position_error_m", _INVALID_NONNEGATIVE_REALS)
def test_bias_training_rows_rejects_invalid_position_gate_before_empty_fast_path(
    max_position_error_m: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_position_error_m must be a finite non-negative real scalar",
    ):
        bias_training_rows(
            pd.DataFrame(),
            pd.DataFrame(),
            source="rf",
            max_position_error_m=max_position_error_m,  # type: ignore[arg-type]
        )


def test_empty_multi_source_fit_still_validates_shared_controls() -> None:
    with pytest.raises(
        ValueError,
        match="ridge_alpha must be a finite non-negative real scalar",
    ):
        fit_bias_correction_models(
            rf=None,
            radar=None,
            truth=pd.DataFrame(),
            ridge_alpha=np.nan,
        )


def test_scalar_like_valid_controls_keep_empty_input_behavior() -> None:
    models = fit_bias_correction_models(
        rf=None,
        radar=None,
        truth=pd.DataFrame(),
        time_gate_s=np.array(0.0),
        ridge_alpha="0",
        min_samples=np.array(1),
    )
    rows = bias_training_rows(
        pd.DataFrame(),
        pd.DataFrame(),
        source="rf",
        max_time_delta_s="0",
        max_position_error_m=np.array(0.0),
    )

    assert models == {}
    assert rows.empty
