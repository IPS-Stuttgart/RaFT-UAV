"""Compatibility fixes for MMUAD candidate-oracle block diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_blocks.py``
module. This package preserves the public import path while treating non-finite
oracle distances as missing candidates instead of silently reporting them as
buried or covered, excluding non-finite values from aggregate error statistics,
and rejecting lossy block-segmentation controls before they can change the
diagnostic policy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _shared_optional_float
from raft_uav.numeric import optional_int as _shared_optional_int

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

_ORIGINAL_BUILD_CANDIDATE_ORACLE_BLOCK_TABLES = _IMPL.build_candidate_oracle_block_tables


def _non_negative_float_control(name: str, value: object) -> float:
    """Return a finite non-negative real scalar control."""

    number = _shared_optional_float(value)
    if number is None or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return number


def _positive_integer_control(name: str, value: object) -> int:
    """Return a positive integer-equivalent scalar control."""

    number = _shared_optional_int(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive integer scalar")
    return number


def build_candidate_oracle_block_tables(
    frame_rows: pd.DataFrame,
    *,
    oracle_error_threshold_m: object = 5.0,
    top_k: object = 5,
    max_gap_s: object = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build block diagnostics after normalizing public policy controls."""

    validated_threshold = _non_negative_float_control(
        "oracle_error_threshold_m",
        oracle_error_threshold_m,
    )
    validated_top_k = _positive_integer_control("top_k", top_k)
    validated_max_gap_s = _non_negative_float_control("max_gap_s", max_gap_s)
    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_BLOCK_TABLES(
        frame_rows,
        oracle_error_threshold_m=validated_threshold,
        top_k=validated_top_k,
        max_gap_s=validated_max_gap_s,
    )


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


_IMPL.build_candidate_oracle_block_tables = build_candidate_oracle_block_tables
_IMPL._failure_mode = _failure_mode
_IMPL._numeric_values = _numeric_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["build_candidate_oracle_block_tables"] = build_candidate_oracle_block_tables
globals()["_non_negative_float_control"] = _non_negative_float_control
globals()["_positive_integer_control"] = _positive_integer_control
globals()["_failure_mode"] = _failure_mode
globals()["_numeric_values"] = _numeric_values

__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
