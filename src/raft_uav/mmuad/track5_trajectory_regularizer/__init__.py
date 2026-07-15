"""Compatibility wrapper for validated Track 5 trajectory regularization.

The maintained implementation lives in the sibling
``track5_trajectory_regularizer.py`` module. This package preserves the public
import path while validating regularizer controls before they can enter the
numerical smoother.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_trajectory_regularizer.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_trajectory_regularizer_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 trajectory-regularizer implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_REGULARIZE = _IMPL.regularize_track5_estimates


def _normalize_finite_float(
    value: Any,
    *,
    field: str,
    allow_zero: bool,
) -> float:
    """Return a finite scalar satisfying the requested sign constraint."""

    qualifier = "non-negative" if allow_zero else "positive"
    message = f"{field} must be a finite {qualifier} scalar"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    invalid_sign = numeric < 0.0 if allow_zero else numeric <= 0.0
    if not np.isfinite(numeric) or invalid_sign:
        raise ValueError(message)
    return numeric


def _normalize_positive_integer(value: Any, *, field: str) -> int:
    """Return a positive finite integer scalar."""

    message = f"{field} must be a positive finite integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def regularize_track5_estimates(
    estimates: pd.DataFrame,
    *,
    smoothness_weight: float = 10.0,
    huber_delta_m: float = 25.0,
    iterations: int = 5,
    observation_sigma_m: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate numerical controls before running the legacy smoother."""

    smoothness_weight = _normalize_finite_float(
        smoothness_weight,
        field="smoothness_weight",
        allow_zero=True,
    )
    huber_delta_m = _normalize_finite_float(
        huber_delta_m,
        field="huber_delta_m",
        allow_zero=False,
    )
    iterations = _normalize_positive_integer(iterations, field="iterations")
    observation_sigma_m = _normalize_finite_float(
        observation_sigma_m,
        field="observation_sigma_m",
        allow_zero=False,
    )
    return _ORIGINAL_REGULARIZE(
        estimates,
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )


_IMPL.regularize_track5_estimates = regularize_track5_estimates

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["regularize_track5_estimates"] = regularize_track5_estimates
globals()["_normalize_finite_float"] = _normalize_finite_float
globals()["_normalize_positive_integer"] = _normalize_positive_integer

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
