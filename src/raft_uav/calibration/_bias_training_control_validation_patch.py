"""Validate bias-training numeric controls before legacy fast paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib

import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_bias = importlib.import_module("raft_uav.calibration.bias")

_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _bias.make_bias_training_examples
_ORIGINAL_FIT_SENSOR_BIAS_CORRECTION = _bias.fit_sensor_bias_correction
_ORIGINAL_FIT_SENSOR_BIAS_CORRECTION_FROM_EXAMPLES = (
    _bias.fit_sensor_bias_correction_from_examples
)
_ORIGINAL_BIAS_TRAINING_ROWS = _bias.bias_training_rows
_ORIGINAL_FIT_BIAS_CORRECTION_MODEL = _bias.fit_bias_correction_model
_ORIGINAL_FIT_BIAS_CORRECTION_BANK = _bias.fit_bias_correction_bank
_ORIGINAL_FIT_BIAS_CORRECTION_MODELS = _bias.fit_bias_correction_models


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    """Return a finite non-Boolean scalar greater than or equal to zero."""

    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    """Return an exact positive non-Boolean integer."""

    normalized = optional_int(value)
    if normalized is None or normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _validated_training_controls(
    *,
    time_gate_s: object,
    ridge_alpha: object,
    min_samples: object,
) -> tuple[float, float, int]:
    return (
        _finite_nonnegative_real(time_gate_s, name="time_gate_s"),
        _finite_nonnegative_real(ridge_alpha, name="ridge_alpha"),
        _positive_integer(min_samples, name="min_samples"),
    )


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Build examples only after validating the nearest-time gate."""

    normalized_time_gate_s = _finite_nonnegative_real(time_gate_s, name="time_gate_s")
    return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
        measurements,
        truth,
        source=source,
        target_columns=target_columns,
        time_gate_s=normalized_time_gate_s,
    )


def fit_sensor_bias_correction_from_examples(
    examples: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
):
    """Fit from examples only after validating all numerical fit controls."""

    normalized_time_gate_s, normalized_ridge_alpha, normalized_min_samples = (
        _validated_training_controls(
            time_gate_s=time_gate_s,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    )
    return _ORIGINAL_FIT_SENSOR_BIAS_CORRECTION_FROM_EXAMPLES(
        examples,
        source=source,
        target_columns=target_columns,
        feature_columns=feature_columns,
        time_gate_s=normalized_time_gate_s,
        ridge_alpha=normalized_ridge_alpha,
        min_samples=normalized_min_samples,
    )


def fit_sensor_bias_correction(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    feature_columns: Sequence[str] | None = None,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
):
    """Fit from measurements only after validating all numerical fit controls."""

    normalized_time_gate_s, normalized_ridge_alpha, normalized_min_samples = (
        _validated_training_controls(
            time_gate_s=time_gate_s,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    )
    return _ORIGINAL_FIT_SENSOR_BIAS_CORRECTION(
        measurements,
        truth,
        source=source,
        target_columns=target_columns,
        feature_columns=feature_columns,
        time_gate_s=normalized_time_gate_s,
        ridge_alpha=normalized_ridge_alpha,
        min_samples=normalized_min_samples,
    )


def bias_training_rows(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float = 2.0,
    max_position_error_m: float | None = None,
) -> pd.DataFrame:
    """Build calibration rows after validating both matching gates."""

    normalized_time_delta = _finite_nonnegative_real(
        max_time_delta_s,
        name="max_time_delta_s",
    )
    normalized_position_error = (
        None
        if max_position_error_m is None
        else _finite_nonnegative_real(
            max_position_error_m,
            name="max_position_error_m",
        )
    )
    return _ORIGINAL_BIAS_TRAINING_ROWS(
        measurements,
        truth,
        source=source,
        max_time_delta_s=normalized_time_delta,
        max_position_error_m=normalized_position_error,
    )


def fit_bias_correction_model(
    rows: pd.DataFrame,
    *,
    source: str,
    ridge_alpha: float = 1.0,
    min_samples: int = 5,
    feature_columns: Sequence[str] | None = None,
):
    """Fit one source model after validating fit controls."""

    normalized_ridge_alpha = _finite_nonnegative_real(ridge_alpha, name="ridge_alpha")
    normalized_min_samples = _positive_integer(min_samples, name="min_samples")
    return _ORIGINAL_FIT_BIAS_CORRECTION_MODEL(
        rows,
        source=source,
        ridge_alpha=normalized_ridge_alpha,
        min_samples=normalized_min_samples,
        feature_columns=feature_columns,
    )


def fit_bias_correction_bank(
    rows_by_source: Mapping[str, pd.DataFrame],
    *,
    ridge_alpha: float = 1.0,
    min_samples: int = 5,
):
    """Validate bank-wide controls even when no source contains training rows."""

    normalized_ridge_alpha = _finite_nonnegative_real(ridge_alpha, name="ridge_alpha")
    normalized_min_samples = _positive_integer(min_samples, name="min_samples")
    return _ORIGINAL_FIT_BIAS_CORRECTION_BANK(
        rows_by_source,
        ridge_alpha=normalized_ridge_alpha,
        min_samples=normalized_min_samples,
    )


def fit_bias_correction_models(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    time_gate_s: float = 2.0,
    ridge_alpha: float = 1.0e-2,
    min_samples: int = 4,
):
    """Validate shared controls even when both source frames are empty."""

    normalized_time_gate_s, normalized_ridge_alpha, normalized_min_samples = (
        _validated_training_controls(
            time_gate_s=time_gate_s,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
        )
    )
    return _ORIGINAL_FIT_BIAS_CORRECTION_MODELS(
        rf=rf,
        radar=radar,
        truth=truth,
        time_gate_s=normalized_time_gate_s,
        ridge_alpha=normalized_ridge_alpha,
        min_samples=normalized_min_samples,
    )


_PATCHED_FUNCTIONS = {
    "make_bias_training_examples": make_bias_training_examples,
    "fit_sensor_bias_correction": fit_sensor_bias_correction,
    "fit_sensor_bias_correction_from_examples": fit_sensor_bias_correction_from_examples,
    "bias_training_rows": bias_training_rows,
    "fit_bias_correction_model": fit_bias_correction_model,
    "fit_bias_correction_bank": fit_bias_correction_bank,
    "fit_bias_correction_models": fit_bias_correction_models,
}

for _name, _function in _PATCHED_FUNCTIONS.items():
    setattr(_bias, _name, _function)
    setattr(_bias._IMPL, _name, _function)
