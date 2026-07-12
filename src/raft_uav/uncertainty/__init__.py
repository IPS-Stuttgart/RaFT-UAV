"""Validation guard for heteroscedastic measurement-uncertainty models.

The maintained implementation remains in the sibling ``uncertainty.py`` module.
This package preserves the public import path while rejecting malformed fitting
controls and persisted variance heads before they can produce invalid covariance
columns.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "uncertainty.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._uncertainty_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load uncertainty implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_PREDICT = _IMPL.VarianceHead.predict
_ORIGINAL_FROM_DICT = _IMPL.VarianceHead.from_dict.__func__
_ORIGINAL_FIT = _IMPL.fit_heteroscedastic_uncertainty_model

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _positive_finite_std(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric


def _nonnegative_finite(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _validated_std_pair(
    min_std_m: object,
    max_std_m: object,
    *,
    name: str,
) -> tuple[float, float]:
    minimum = _positive_finite_std(min_std_m, name=f"{name} min_std_m")
    maximum = _positive_finite_std(max_std_m, name=f"{name} max_std_m")
    if minimum > maximum:
        raise ValueError(f"{name} min_std_m must not exceed max_std_m")
    return minimum, maximum


def _validated_std_maps(
    min_std_m: Mapping[str, Mapping[str, float]] | None,
    max_std_m: Mapping[str, Mapping[str, float]] | None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    minimum = _IMPL._nested(_IMPL.DEFAULT_MIN_STD, min_std_m)
    maximum = _IMPL._nested(_IMPL.DEFAULT_MAX_STD, max_std_m)
    for source, dimensions in _IMPL.SOURCE_DIMS.items():
        for dimension in dimensions:
            low, high = _validated_std_pair(
                minimum[source][dimension],
                maximum[source][dimension],
                name=f"{source}.{dimension}",
            )
            minimum[source][dimension] = low
            maximum[source][dimension] = high
    return minimum, maximum


def _predict_with_validated_bounds(self: Any, frame: pd.DataFrame) -> np.ndarray:
    _validated_std_pair(
        self.min_std_m,
        self.max_std_m,
        name=f"{self.source}.{self.dimension}",
    )
    return _ORIGINAL_PREDICT(self, frame)


def _variance_head_from_dict(cls: type[Any], item: Mapping[str, Any]) -> Any:
    head = _ORIGINAL_FROM_DICT(cls, item)
    _validated_std_pair(
        head.min_std_m,
        head.max_std_m,
        name=f"{head.source}.{head.dimension}",
    )
    return head


def fit_heteroscedastic_uncertainty_model(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    ridge_lambda: float = 1.0,
    max_time_delta_s: float = 2.0,
    min_std_m: Mapping[str, Mapping[str, float]] | None = None,
    max_std_m: Mapping[str, Mapping[str, float]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Fit a model after validating all numerical controls and variance bounds."""

    ridge = _nonnegative_finite(ridge_lambda, name="ridge_lambda")
    time_gate = _nonnegative_finite(max_time_delta_s, name="max_time_delta_s")
    minimum, maximum = _validated_std_maps(min_std_m, max_std_m)
    return _ORIGINAL_FIT(
        rf=rf,
        radar=radar,
        truth=truth,
        ridge_lambda=ridge,
        max_time_delta_s=time_gate,
        min_std_m=minimum,
        max_std_m=maximum,
        metadata=metadata,
    )


_IMPL.VarianceHead.predict = _predict_with_validated_bounds
_IMPL.VarianceHead.from_dict = classmethod(_variance_head_from_dict)
_IMPL.fit_heteroscedastic_uncertainty_model = fit_heteroscedastic_uncertainty_model

VarianceHead = _IMPL.VarianceHead
HeteroscedasticUncertaintyModel = _IMPL.HeteroscedasticUncertaintyModel

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
