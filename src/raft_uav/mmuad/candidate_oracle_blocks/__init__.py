"""Compatibility fixes for MMUAD candidate-oracle block diagnostics.

The maintained implementation lives in the sibling ``candidate_oracle_blocks.py``
module. This package preserves the public import path while treating non-finite
oracle distances as missing candidates instead of silently reporting them as
buried or covered, and while excluding non-finite values from aggregate error
statistics.
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


_IMPL._failure_mode = _failure_mode
_IMPL._numeric_values = _numeric_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_failure_mode"] = _failure_mode
globals()["_numeric_values"] = _numeric_values

__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
