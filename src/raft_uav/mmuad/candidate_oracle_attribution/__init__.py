"""Compatibility validation for candidate-oracle attribution controls.

The maintained implementation lives in the sibling
``candidate_oracle_attribution.py`` module. This package preserves the public
import path while rejecting malformed truth-matching time gates and top-K values
before they can silently widen, empty, or change the diagnostic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_attribution.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_attribution_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-oracle attribution implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_TABLES = _IMPL.build_candidate_oracle_attribution_tables


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a validated non-negative finite scalar control."""

    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if array.ndim != 0 or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a non-negative finite scalar")
    try:
        normalized = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative finite scalar") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a non-negative finite scalar")
    return normalized


def _normalize_top_k_values(values: Sequence[object]) -> tuple[int, ...]:
    """Return sorted unique positive integer top-K values without truncation."""

    message = "top_k_values must contain only positive integers"
    if isinstance(values, (str, bytes)):
        raise ValueError(message)
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(message) from exc

    normalized: list[int] = []
    for value in raw_values:
        try:
            number = _nonnegative_finite_scalar(value, name="top_k_values")
        except ValueError as exc:
            raise ValueError(message) from exc
        if number <= 0.0 or not number.is_integer():
            raise ValueError(message)
        normalized.append(int(number))
    return tuple(sorted(set(normalized)))


def build_candidate_oracle_attribution_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k_values: Sequence[int] = _IMPL._DEFAULT_TOP_K,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "ranker_score",
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return oracle-attribution tables with validated scalar controls."""

    max_delta = _nonnegative_finite_scalar(
        max_truth_time_delta_s,
        name="max_truth_time_delta_s",
    )
    top_k = _normalize_top_k_values(top_k_values)
    return _ORIGINAL_BUILD_TABLES(
        candidates,
        truth,
        top_k_values=top_k,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        max_truth_time_delta_s=max_delta,
    )


_IMPL.build_candidate_oracle_attribution_tables = (
    build_candidate_oracle_attribution_tables
)
_IMPL._normalize_top_k_values = _normalize_top_k_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_top_k_values"] = _normalize_top_k_values
globals()["build_candidate_oracle_attribution_tables"] = (
    build_candidate_oracle_attribution_tables
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
