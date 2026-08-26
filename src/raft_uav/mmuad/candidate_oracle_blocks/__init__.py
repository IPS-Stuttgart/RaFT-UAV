"""Compatibility fixes for MMUAD candidate-oracle block diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_blocks.py``
module. This package preserves the public import path while treating non-finite
oracle distances as missing candidates instead of silently reporting them as
buried or covered, excluding non-finite values from aggregate error statistics,
and validating block-construction controls before they can silently change
diagnostic semantics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_blocks.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_blocks_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load candidate-oracle block implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_BUILD = _IMPL.build_candidate_oracle_block_tables


def _failure_mode(
    rows: pd.DataFrame,
    *,
    top_k_column: str,
    oracle_error_threshold_m: float,
) -> pd.Series:
    """Classify invalid oracle distances as missing candidate evidence."""

    errors = pd.to_numeric(rows["oracle_all_3d_m"], errors="coerce")
    invalid_error = pd.Series(
        ~np.isfinite(errors.to_numpy(float)),
        index=rows.index,
        dtype=bool,
    )
    missing_candidate = invalid_error | errors.gt(float(oracle_error_threshold_m))
    buried_candidate = (~missing_candidate) & (~rows[top_k_column])
    return pd.Series(
        np.select(
            [missing_candidate, buried_candidate],
            ["missing_good_candidate", "good_candidate_buried"],
            default="covered_in_topk",
        ),
        index=rows.index,
    )


def _numeric_values(values: Any) -> pd.Series:
    """Return only finite numeric values for block summary statistics."""

    if values is None:
        return pd.Series(dtype=float)
    if isinstance(values, pd.Series):
        raw = values
    elif np.isscalar(values):
        raw = pd.Series([values])
    else:
        raw = pd.Series(values)
    numeric = pd.to_numeric(raw, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(float))
    return numeric.loc[finite]


def _unwrap_scalar(value: Any, *, name: str) -> Any:
    """Return a Python-like scalar while rejecting ambiguous containers."""

    if isinstance(value, np.ma.MaskedArray):
        if value.ndim != 0 or np.ma.is_masked(value):
            raise ValueError(f"{name} must be a finite scalar")
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a finite scalar")
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA:
        raise ValueError(f"{name} must be a finite scalar")
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be Boolean")
    return value


def _finite_nonnegative_real(value: Any, *, name: str) -> float:
    """Parse a finite non-negative real scalar without Boolean truthiness."""

    value = _unwrap_scalar(value, name=name)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative real scalar") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_integer(value: Any, *, name: str) -> int:
    """Parse a positive integer-equivalent scalar without lossy truncation."""

    value = _unwrap_scalar(value, name=name)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(parsed) or not parsed.is_integer() or parsed <= 0.0:
        raise ValueError(f"{name} must be a positive integer")
    integer = int(parsed)
    if float(integer) != parsed:
        raise ValueError(f"{name} must be exactly representable as an integer")
    return integer


def build_candidate_oracle_block_tables(
    frame_rows: pd.DataFrame,
    *,
    oracle_error_threshold_m: float = 5.0,
    top_k: int = 5,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate-oracle blocks with strict diagnostic controls."""

    threshold = _finite_nonnegative_real(
        oracle_error_threshold_m,
        name="oracle_error_threshold_m",
    )
    parsed_top_k = _positive_integer(top_k, name="top_k")
    gap = _finite_nonnegative_real(max_gap_s, name="max_gap_s")
    return _ORIGINAL_BUILD(
        frame_rows,
        oracle_error_threshold_m=threshold,
        top_k=parsed_top_k,
        max_gap_s=gap,
    )


_IMPL._failure_mode = _failure_mode
_IMPL._numeric_values = _numeric_values
_IMPL.build_candidate_oracle_block_tables = build_candidate_oracle_block_tables

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_failure_mode"] = _failure_mode
globals()["_numeric_values"] = _numeric_values
globals()["_unwrap_scalar"] = _unwrap_scalar
globals()["_finite_nonnegative_real"] = _finite_nonnegative_real
globals()["_positive_integer"] = _positive_integer
globals()["build_candidate_oracle_block_tables"] = build_candidate_oracle_block_tables

__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
