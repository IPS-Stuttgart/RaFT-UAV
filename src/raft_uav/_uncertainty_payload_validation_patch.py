"""Reject malformed uncertainty payloads and non-finite fit residuals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


def _finite_real_scalar(
    value: object,
    *,
    name: str,
    nonfinite_error: str | None = None,
) -> float:
    """Return a finite real scalar without Boolean or array coercion."""

    error = f"{name} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0:
        raise ValueError(error)
    if np.iscomplexobj(scalar):
        raise ValueError(nonfinite_error or error)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(nonfinite_error or error)
    return number


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return a nonnegative integer scalar without truncation."""

    number = _finite_real_scalar(value, name=name)
    if number < 0.0 or not number.is_integer():
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(number)


def _validated_coefficients(values: object) -> tuple[object, ...]:
    """Validate raw coefficient values before legacy float coercion."""

    if isinstance(values, (str, bytes)):
        raise ValueError("coefficients must be a sequence of finite real scalars")
    try:
        coefficients = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "coefficients must be a sequence of finite real scalars"
        ) from exc
    for index, value in enumerate(coefficients):
        _finite_real_scalar(
            value,
            name=f"coefficients[{index}]",
            nonfinite_error="variance head coefficients must be finite numbers",
        )
    return coefficients


def _validate_head_payload(item: object) -> Mapping[str, Any]:
    """Validate raw serialized fields before the legacy loader coerces them."""

    if not isinstance(item, Mapping):
        raise ValueError("uncertainty variance head must be a mapping")
    _validated_coefficients(item.get("coefficients", ()))
    _finite_real_scalar(item.get("min_std_m"), name="min_std_m")
    _finite_real_scalar(item.get("max_std_m"), name="max_std_m")
    _nonnegative_integer(item.get("training_rows", 0), name="training_rows")
    return item


def _validate_model_payload(payload: object) -> Mapping[str, Any]:
    """Reject malformed top-level containers before legacy coercion."""

    if not isinstance(payload, Mapping):
        raise ValueError("uncertainty model payload must be a mapping")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("uncertainty model metadata must be a mapping")

    if "heads" in payload:
        heads = payload["heads"]
        if isinstance(heads, (str, bytes, bytearray, Mapping)) or not isinstance(
            heads, Iterable
        ):
            raise ValueError("uncertainty model heads must be an iterable of mappings")

    return payload


def _mask_nonfinite_fit_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite residuals with missing values before variance fitting."""

    residual_columns = [
        column
        for column in frame.columns
        if column.startswith("residual_") and column.endswith("_m")
    ]
    if frame.empty or not residual_columns:
        return frame

    out = frame.copy()
    for column in residual_columns:
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        nonfinite = ~np.isfinite(values)
        if bool(np.any(nonfinite)):
            out.loc[nonfinite, column] = np.nan
    return out


def install() -> None:
    """Install raw-field, container, and fit-residual validation."""

    from raft_uav import uncertainty as uncertainty_module

    head_class = uncertainty_module.VarianceHead
    if not getattr(head_class, "_raft_uav_raw_field_validation_installed", False):
        original_init = head_class.__init__
        original_from_dict = head_class.from_dict.__func__

        def validated_init(
            self,
            source,
            dimension,
            feature_names,
            coefficients,
            min_std_m,
            max_std_m,
            training_rows,
        ):
            _validated_coefficients(coefficients)
            _nonnegative_integer(training_rows, name="training_rows")
            original_init(
                self,
                source,
                dimension,
                feature_names,
                coefficients,
                min_std_m,
                max_std_m,
                training_rows,
            )

        def validated_from_dict(cls, item):
            _validate_head_payload(item)
            return original_from_dict(cls, item)

        head_class.__init__ = validated_init
        head_class.from_dict = classmethod(validated_from_dict)
        head_class._raft_uav_raw_field_validation_installed = True

    model_class = uncertainty_module.HeteroscedasticUncertaintyModel
    if not getattr(model_class, "_raft_uav_container_validation_installed", False):
        original_model_from_dict = model_class.from_dict.__func__

        def validated_model_from_dict(cls, payload):
            _validate_model_payload(payload)
            return original_model_from_dict(cls, payload)

        model_class.from_dict = classmethod(validated_model_from_dict)
        model_class._raft_uav_container_validation_installed = True

    if getattr(
        uncertainty_module,
        "_raft_uav_nonfinite_fit_residual_validation_installed",
        False,
    ):
        return

    legacy = uncertainty_module._legacy
    original_aligned_residuals = legacy._aligned_residuals

    @wraps(original_aligned_residuals)
    def validated_aligned_residuals(*args, **kwargs):
        aligned = original_aligned_residuals(*args, **kwargs)
        return _mask_nonfinite_fit_residuals(aligned)

    legacy._aligned_residuals = validated_aligned_residuals
    uncertainty_module._aligned_residuals = validated_aligned_residuals
    uncertainty_module._raft_uav_nonfinite_fit_residual_validation_installed = True
