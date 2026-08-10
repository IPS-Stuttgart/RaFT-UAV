"""Compatibility wrapper for robust candidate-diversity inputs.

The maintained implementation lives in the sibling ``candidate_diversity.py``
module. This package preserves the public import path while coercing candidate
coordinates before spatial filtering so malformed rows are skipped rather than
raising during NumPy conversion. It also validates diversity controls before
legacy numeric coercion can silently clamp or truncate them, normalizes score
columns row by row without discarding complex components, and rejects complex
or non-binary numeric protected-candidate flags before they can be treated as
truthy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_diversity.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_diversity_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-diversity implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_DIVERSIFY = _IMPL.diversify_candidate_reservoir
_ORIGINAL_PARSE_PROTECTED_FLAG = _IMPL._parse_protected_flag
_TRUE_BOOLEAN_TEXT = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TEXT = frozenset({"0", "false", "f", "no", "n", "off"})


def _scalar_value(value: Any, *, message: str) -> Any:
    """Recursively unwrap zero-dimensional scalar containers."""

    scalar = value
    seen_array_ids: set[int] = set()
    while isinstance(scalar, np.ndarray):
        if np.ma.is_masked(scalar) or scalar.ndim != 0:
            raise ValueError(message)
        array_id = id(scalar)
        if array_id in seen_array_ids:
            raise ValueError(message)
        seen_array_ids.add(array_id)
        scalar = scalar.item()
    if np.ma.is_masked(scalar):
        raise ValueError(message)
    return scalar


def _finite_nonnegative_control(value: Any, *, name: str) -> float:
    """Return a finite nonnegative real scalar without Boolean coercion."""

    message = f"{name} must be a finite non-negative number"
    scalar = _scalar_value(value, message=message)
    if isinstance(scalar, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _finite_positive_control(value: Any, *, name: str) -> float:
    """Return a finite positive real scalar without Boolean coercion."""

    number = _finite_nonnegative_control(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _positive_integer_control(value: Any, *, name: str) -> int:
    """Return a positive integer scalar without fractional truncation."""

    message = f"{name} must be a positive integer"
    scalar = _scalar_value(value, message=message)
    if isinstance(scalar, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 1.0 or not number.is_integer():
        raise ValueError(message)
    return int(number)


def _boolean_control(value: Any, *, name: str) -> bool:
    """Return one explicit Boolean control without string truthiness."""

    message = f"{name} must be a Boolean scalar"
    scalar = _scalar_value(value, message=message)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, str):
        text = scalar.strip().casefold()
        if text in _TRUE_BOOLEAN_TEXT:
            return True
        if text in _FALSE_BOOLEAN_TEXT:
            return False
        raise ValueError(message)
    if isinstance(scalar, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number not in {0.0, 1.0}:
        raise ValueError(message)
    return bool(number)


def _finite_real_score(value: Any) -> float:
    """Return a finite real score or NaN for malformed scalar values."""

    if value is None or np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return float("nan")
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError):
        return float("nan")
    if scalar.ndim != 0:
        return float("nan")
    try:
        item = scalar.item()
    except (TypeError, ValueError):
        return float("nan")
    if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
        return float("nan")
    if isinstance(item, (complex, np.complexfloating)):
        if not np.isfinite(item.real) or not np.isfinite(item.imag) or item.imag != 0.0:
            return float("nan")
        return float(item.real)
    try:
        number = float(item)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _finite_real_score_series(values: pd.Series) -> pd.Series:
    """Normalize scores row by row without complex-to-real truncation."""

    return values.map(_finite_real_score).astype(float)


def _parse_protected_flag(value: Any) -> bool:
    """Reject complex and non-binary numeric flags before legacy coercion."""

    if not np.ma.is_masked(value):
        try:
            scalar = np.asarray(value)
        except (TypeError, ValueError):
            scalar = None
        if scalar is not None and scalar.ndim == 0:
            try:
                item = scalar.item()
            except (TypeError, ValueError):
                item = value
            if isinstance(item, (complex, np.complexfloating)):
                raise ValueError(
                    "candidate_reservoir_protected values must be boolean-like; "
                    f"got {value!r}"
                )
            if not isinstance(item, (bool, np.bool_, str)) and item is not None:
                try:
                    missing = pd.isna(item)
                except (TypeError, ValueError):
                    missing = False
                if not (isinstance(missing, (bool, np.bool_)) and bool(missing)):
                    try:
                        number = float(item)
                    except (TypeError, ValueError, OverflowError):
                        number = float("nan")
                    if np.isfinite(number) and number not in {0.0, 1.0}:
                        raise ValueError(
                            "candidate_reservoir_protected values must be boolean-like; "
                            f"got {value!r}"
                        )
    return _ORIGINAL_PARSE_PROTECTED_FLAG(value)


def diversify_candidate_reservoir(rows, **kwargs):
    """Validate controls and coerce coordinates and scores before pruning."""

    normalized_kwargs = dict(kwargs)
    normalized_kwargs["radius_m"] = _finite_nonnegative_control(
        normalized_kwargs.get("radius_m", 1.0),
        name="radius_m",
    )
    normalized_kwargs["max_candidates_per_frame"] = _positive_integer_control(
        normalized_kwargs.get("max_candidates_per_frame", 40),
        name="max_candidates_per_frame",
    )
    normalized_kwargs["preserve_protected"] = _boolean_control(
        normalized_kwargs.get("preserve_protected", True),
        name="preserve_protected",
    )
    normalized_kwargs["uncertainty_reference_m"] = _finite_positive_control(
        normalized_kwargs.get("uncertainty_reference_m", 10.0),
        name="uncertainty_reference_m",
    )
    normalized_kwargs["uncertainty_exponent"] = _finite_nonnegative_control(
        normalized_kwargs.get("uncertainty_exponent", 0.5),
        name="uncertainty_exponent",
    )
    normalized_kwargs["min_radius_scale"] = _finite_positive_control(
        normalized_kwargs.get("min_radius_scale", 0.25),
        name="min_radius_scale",
    )
    normalized_kwargs["max_radius_scale"] = _finite_positive_control(
        normalized_kwargs.get("max_radius_scale", 4.0),
        name="max_radius_scale",
    )

    frame = pd.DataFrame(rows).copy()
    score_column = normalized_kwargs.get("score_column", "candidate_reservoir_score")
    if score_column in frame.columns:
        frame[score_column] = _finite_real_score_series(frame[score_column])
    if "confidence" in frame.columns:
        frame["confidence"] = _finite_real_score_series(frame["confidence"])
    for column in ("x_m", "y_m", "z_m"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _ORIGINAL_DIVERSIFY(frame, **normalized_kwargs)


_IMPL._scalar_value = _scalar_value
_IMPL._finite_nonnegative_control = _finite_nonnegative_control
_IMPL._finite_positive_control = _finite_positive_control
_IMPL._positive_integer_control = _positive_integer_control
_IMPL._boolean_control = _boolean_control
_IMPL._finite_real_score = _finite_real_score
_IMPL._finite_real_score_series = _finite_real_score_series
_IMPL._parse_protected_flag = _parse_protected_flag
_IMPL.diversify_candidate_reservoir = diversify_candidate_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_scalar_value"] = _scalar_value
globals()["_finite_nonnegative_control"] = _finite_nonnegative_control
globals()["_finite_positive_control"] = _finite_positive_control
globals()["_positive_integer_control"] = _positive_integer_control
globals()["_boolean_control"] = _boolean_control
globals()["_finite_real_score"] = _finite_real_score
globals()["_finite_real_score_series"] = _finite_real_score_series
globals()["_parse_protected_flag"] = _parse_protected_flag
globals()["diversify_candidate_reservoir"] = diversify_candidate_reservoir

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
