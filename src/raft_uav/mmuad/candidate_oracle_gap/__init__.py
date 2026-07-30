"""Compatibility package for validated MMUAD candidate oracle-gap diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_gap.py``
module. This package validates the nearest-time gate and prevents genuinely
complex timestamps, positions, or confidence values from being silently cast
to their real components.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_oracle_gap.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_oracle_gap_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy candidate oracle-gap module from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP = _IMPL.build_candidate_oracle_gap
_ORIGINAL_FINITE_CANDIDATE_ROWS = _IMPL._finite_candidate_rows
_ORIGINAL_FINITE_TRUTH_ROWS = _IMPL._finite_truth_rows


def _coerce_real_numeric_columns(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Convert numeric columns without discarding nonzero imaginary components."""

    normalized = rows.copy()
    for column in columns:
        if column not in normalized.columns:
            continue
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        values = numeric.to_numpy()
        if np.iscomplexobj(values):
            real = np.real(values)
            imaginary = np.imag(values)
            numeric = pd.Series(
                np.where(
                    np.isfinite(imaginary) & (imaginary == 0.0),
                    real,
                    np.nan,
                ),
                index=normalized.index,
                dtype=float,
            )
        normalized[column] = numeric
    return normalized


def _finite_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Reject candidates whose required numeric fields are genuinely complex."""

    normalized = _coerce_real_numeric_columns(
        candidates,
        ("time_s", "x_m", "y_m", "z_m", "confidence"),
    )
    return _ORIGINAL_FINITE_CANDIDATE_ROWS(normalized)


def _finite_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    """Reject truth rows whose timestamps or positions are genuinely complex."""

    normalized = _coerce_real_numeric_columns(
        truth,
        ("time_s", "x_m", "y_m", "z_m"),
    )
    return _ORIGINAL_FINITE_TRUTH_ROWS(normalized)


def _normalize_max_time_delta_s(value: Any) -> float | None:
    if value is None:
        return None
    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(
            "max_time_delta_s must be a nonnegative finite scalar or None"
        )
    return normalized


@wraps(_ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP)
def build_candidate_oracle_gap(
    candidates: Any,
    selected: Any,
    truth: Any,
    *,
    max_time_delta_s: float | None = 0.5,
) -> Any:
    """Build oracle-gap rows after validating the nearest-time gate."""

    return _ORIGINAL_BUILD_CANDIDATE_ORACLE_GAP(
        candidates,
        selected,
        truth,
        max_time_delta_s=_normalize_max_time_delta_s(max_time_delta_s),
    )


_IMPL._finite_candidate_rows = _finite_candidate_rows
_IMPL._finite_truth_rows = _finite_truth_rows
_IMPL.build_candidate_oracle_gap = build_candidate_oracle_gap

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_coerce_real_numeric_columns"] = _coerce_real_numeric_columns
globals()["_finite_candidate_rows"] = _finite_candidate_rows
globals()["_finite_truth_rows"] = _finite_truth_rows
globals()["_normalize_max_time_delta_s"] = _normalize_max_time_delta_s
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
