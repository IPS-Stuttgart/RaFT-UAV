"""Compatibility package validating learned candidate-uncertainty models.

The maintained implementation lives in the sibling ``candidate_uncertainty.py``
module. This package preserves the public import path while rejecting malformed
training controls and saved-model payloads before they can produce non-finite or
shape-dependent uncertainty predictions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_uncertainty.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_uncertainty_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate uncertainty implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRAIN = _IMPL.train_candidate_uncertainty
_ORIGINAL_PREDICT = _IMPL.predict_candidate_sigma
_ORIGINAL_APPLY = _IMPL.apply_candidate_uncertainty
_ORIGINAL_SAVE = _IMPL.save_candidate_uncertainty_model
_ORIGINAL_LOAD = _IMPL.load_candidate_uncertainty_model


def _finite_float(value: Any, *, name: str) -> float:
    """Return one finite non-Boolean scalar with a field-specific error."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be finite")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _exact_integer(value: Any, *, name: str, positive: bool = False) -> int:
    """Return an exact integer without truncating floats or accepting Booleans."""

    number = _finite_float(value, name=name)
    if not number.is_integer():
        qualifier = "a positive integer" if positive else "an integer"
        raise ValueError(f"{name} must be {qualifier}")
    integer = int(number)
    if positive and integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _model_vector(model: Any, name: str, expected_size: int) -> np.ndarray:
    """Return one finite one-dimensional model vector of the expected length."""

    try:
        values = np.asarray(getattr(model, name), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {expected_size} finite values") from exc
    if values.ndim != 1 or values.size != expected_size or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {expected_size} finite values")
    return values


def _validate_candidate_uncertainty_model(model: Any) -> Any:
    """Validate structural and numeric invariants of a portable uncertainty model."""

    if str(model.model_type) not in _IMPL.MODEL_TYPES:
        raise ValueError(f"unsupported uncertainty model_type={model.model_type!r}")
    if str(model.target_transform) not in _IMPL.TARGET_TRANSFORMS:
        raise ValueError(f"unsupported target_transform={model.target_transform!r}")

    feature_columns = [str(value) for value in model.feature_columns]
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must be unique")

    feature_count = len(feature_columns)
    _model_vector(model, "feature_means", feature_count)
    scales = _model_vector(model, "feature_scales", feature_count)
    if np.any(scales <= 0.0):
        raise ValueError("feature_scales must contain positive finite values")
    _model_vector(model, "weights", feature_count)

    sigma_min = _finite_float(model.sigma_min_m, name="sigma_min_m")
    sigma_max = _finite_float(model.sigma_max_m, name="sigma_max_m")
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min_m <= sigma_max_m")
    fallback = _finite_float(model.fallback_sigma_m, name="fallback_sigma_m")
    if not sigma_min <= fallback <= sigma_max:
        raise ValueError("fallback_sigma_m must lie within the sigma bounds")
    _finite_float(model.bias, name="bias")

    if model.model_type != "ridge":
        payload = model.sklearn_estimator_base64
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError(
                f"{model.model_type} uncertainty model requires sklearn_estimator_base64"
            )
    return model


def train_candidate_uncertainty(
    features,
    *,
    model_type: str = "hist-gradient-boosting",
    target_transform: str = "log1p",
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 30.0,
    ridge_alpha: float = 1.0,
    random_state: int = 13,
    n_estimators: int = 300,
):
    """Train after validating controls that legacy comparisons can let through."""

    sigma_min = _finite_float(sigma_min_m, name="sigma_min_m")
    sigma_max = _finite_float(sigma_max_m, name="sigma_max_m")
    if sigma_min <= 0.0 or sigma_max < sigma_min:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min_m <= sigma_max_m")

    normalized_model_type = str(model_type)
    normalized_ridge_alpha = ridge_alpha
    normalized_random_state = random_state
    normalized_n_estimators = n_estimators
    if normalized_model_type == "ridge":
        normalized_ridge_alpha = _finite_float(ridge_alpha, name="ridge_alpha")
        if normalized_ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and non-negative")
    elif normalized_model_type in {"random-forest", "hist-gradient-boosting"}:
        normalized_random_state = _exact_integer(random_state, name="random_state")
        normalized_n_estimators = _exact_integer(
            n_estimators,
            name="n_estimators",
            positive=True,
        )

    model = _ORIGINAL_TRAIN(
        features,
        model_type=normalized_model_type,
        target_transform=target_transform,
        sigma_min_m=sigma_min,
        sigma_max_m=sigma_max,
        ridge_alpha=normalized_ridge_alpha,
        random_state=normalized_random_state,
        n_estimators=normalized_n_estimators,
    )
    return _validate_candidate_uncertainty_model(model)


def predict_candidate_sigma(features, model):
    """Predict only from a structurally valid portable model."""

    return _ORIGINAL_PREDICT(
        features,
        _validate_candidate_uncertainty_model(model),
    )


def apply_candidate_uncertainty(
    candidates,
    model,
    *,
    output_column: str = "predicted_sigma_m",
    replace_covariance: bool = False,
    z_scale: float = 1.0,
):
    """Apply a valid model and reject invalid covariance scaling controls."""

    normalized_z_scale = z_scale
    if replace_covariance:
        try:
            normalized_z_scale = _finite_float(z_scale, name="z_scale")
        except ValueError as exc:
            raise ValueError("z_scale must be finite and positive") from exc
        if normalized_z_scale <= 0.0:
            raise ValueError("z_scale must be finite and positive")
    return _ORIGINAL_APPLY(
        candidates,
        _validate_candidate_uncertainty_model(model),
        output_column=output_column,
        replace_covariance=replace_covariance,
        z_scale=normalized_z_scale,
    )


def save_candidate_uncertainty_model(model, path: Path) -> Path:
    """Write only structurally valid portable model payloads."""

    return _ORIGINAL_SAVE(_validate_candidate_uncertainty_model(model), path)


def load_candidate_uncertainty_model(path: Path):
    """Read and validate a portable model before exposing it to inference."""

    return _validate_candidate_uncertainty_model(_ORIGINAL_LOAD(path))


_IMPL._validate_candidate_uncertainty_model = _validate_candidate_uncertainty_model
_IMPL.train_candidate_uncertainty = train_candidate_uncertainty
_IMPL.predict_candidate_sigma = predict_candidate_sigma
_IMPL.apply_candidate_uncertainty = apply_candidate_uncertainty
_IMPL.save_candidate_uncertainty_model = save_candidate_uncertainty_model
_IMPL.load_candidate_uncertainty_model = load_candidate_uncertainty_model

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_float"] = _finite_float
globals()["_exact_integer"] = _exact_integer
globals()["_model_vector"] = _model_vector
globals()["_validate_candidate_uncertainty_model"] = (
    _validate_candidate_uncertainty_model
)
globals()["train_candidate_uncertainty"] = train_candidate_uncertainty
globals()["predict_candidate_sigma"] = predict_candidate_sigma
globals()["apply_candidate_uncertainty"] = apply_candidate_uncertainty
globals()["save_candidate_uncertainty_model"] = save_candidate_uncertainty_model
globals()["load_candidate_uncertainty_model"] = load_candidate_uncertainty_model

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
