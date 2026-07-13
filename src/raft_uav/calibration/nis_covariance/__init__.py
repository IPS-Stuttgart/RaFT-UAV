"""Compatibility wrapper hardening NIS diagnostics normalization.

The maintained implementation lives in the sibling ``nis_covariance.py``
module. This package preserves the public import path while rejecting malformed
diagnostics rows before they can create bogus calibration groups or fail during
measurement-dimension conversion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "nis_covariance.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._nis_covariance_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load NIS covariance implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _truthy(value: object) -> bool:
    """Interpret only explicit non-missing values as accepted diagnostics."""

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool | np.bool_) and bool(missing):
        return False
    return bool(value)


def _normalized_diagnostics_frame(
    frame: pd.DataFrame,
    *,
    accepted_only: bool,
) -> pd.DataFrame:
    """Drop malformed diagnostics without inventing or truncating group keys."""

    required = {"source", "measurement_dim", "nis"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"diagnostics frame is missing required columns: {missing}")

    work = frame.copy()
    if accepted_only and "accepted" in work.columns:
        work = work.loc[work["accepted"].map(_truthy)].copy()

    source = work["source"].astype("string").str.strip()
    measurement_dim = pd.to_numeric(work["measurement_dim"], errors="coerce")
    nis = pd.to_numeric(work["nis"], errors="coerce")
    dim_values = measurement_dim.to_numpy(dtype=float)
    nis_values = nis.to_numpy(dtype=float)
    valid = (
        source.notna().to_numpy(dtype=bool)
        & source.ne("").fillna(False).to_numpy(dtype=bool)
        & np.isfinite(dim_values)
        & (dim_values > 0.0)
        & (dim_values < float(2**63))
        & np.equal(dim_values, np.floor(dim_values))
        & np.isfinite(nis_values)
        & (nis_values >= 0.0)
    )
    work = work.loc[valid].copy()
    work["source"] = source.loc[work.index].astype(str)
    work["measurement_dim"] = measurement_dim.loc[work.index].astype(int)
    work["nis"] = nis.loc[work.index].astype(float)
    return work


_IMPL._truthy = _truthy
_IMPL._normalized_diagnostics_frame = _normalized_diagnostics_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_truthy"] = _truthy
globals()["_normalized_diagnostics_frame"] = _normalized_diagnostics_frame
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
