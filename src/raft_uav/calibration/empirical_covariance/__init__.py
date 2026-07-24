"""Compatibility validation for empirical covariance sequence alignment.

The maintained implementation lives in the sibling ``empirical_covariance.py``
module. This package preserves the public import path while rejecting one-sided
sequence metadata before timestamp-only alignment can mix unrelated flights.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "empirical_covariance.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._empirical_covariance_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import guard
    raise ImportError(
        f"cannot load empirical covariance implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ALIGNED_RESIDUALS = _IMPL.aligned_residuals


def aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
):
    """Align residuals only when sequence metadata is structurally consistent."""

    frame_has_sequence = "sequence_id" in frame.columns
    truth_has_sequence = "sequence_id" in truth.columns
    if frame_has_sequence != truth_has_sequence:
        raise ValueError(
            "frame and truth must either both contain sequence_id or both omit it"
        )
    return _ORIGINAL_ALIGNED_RESIDUALS(
        frame,
        truth,
        source=source,
        max_time_delta_s=max_time_delta_s,
    )


_IMPL.aligned_residuals = aligned_residuals

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_ALIGNED_RESIDUALS"] = _ORIGINAL_ALIGNED_RESIDUALS
globals()["aligned_residuals"] = aligned_residuals

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
