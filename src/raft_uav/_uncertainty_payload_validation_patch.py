"""Reject malformed uncertainty payloads and non-finite fit coordinates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

_COORDINATE_COLUMNS = ("east_m", "north_m", "up_m")


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
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
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


def _mask_nonfinite_fit_coordinates(
    frame: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Mask non-finite coordinates so they cannot become clipped fit targets."""

    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    for column in _COORDINATE_COLUMNS:
        if column not in out.columns:
            continue
        try:
            values = pd.to_numeric(out[column], errors="raise")
            raw = values.to_numpy()
            if np.iscomplexobj(raw):
                continue
            numeric = values.to_numpy(dtype=float, na_value=np.nan)
        except (TypeError, ValueError, OverflowError):
            continue
        nonfinite = ~np.isfinite(numeric)
        if bool(np.any(nonfinite)):
            out.loc[nonfinite, column] = np.nan
    return out


def install() -> None:
    """Install raw-field, container, and fit-coordinate validation."""

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
        "_raft_uav_nonfinite_fit_coordinate_validation_installed",
        False,
    ):
        return

    original_fit = uncertainty_module.fit_heteroscedastic_uncertainty_model

    def validated_fit_heteroscedastic_uncertainty_model(
        *,
        rf,
        radar,
        truth,
        ridge_lambda=1.0,
        max_time_delta_s=2.0,
        min_std_m=None,
        max_std_m=None,
        metadata=None,
    ):
        return original_fit(
            rf=_mask_nonfinite_fit_coordinates(rf),
            radar=_mask_nonfinite_fit_coordinates(radar),
            truth=_mask_nonfinite_fit_coordinates(truth),
            ridge_lambda=ridge_lambda,
            max_time_delta_s=max_time_delta_s,
            min_std_m=min_std_m,
            max_std_m=max_std_m,
            metadata=metadata,
        )

    uncertainty_module.fit_heteroscedastic_uncertainty_model = (
        validated_fit_heteroscedastic_uncertainty_model
    )
    legacy = getattr(uncertainty_module, "_legacy", None)
    if legacy is not None:
        legacy.fit_heteroscedastic_uncertainty_model = (
            validated_fit_heteroscedastic_uncertainty_model
        )
    uncertainty_module._raft_uav_nonfinite_fit_coordinate_validation_installed = True
