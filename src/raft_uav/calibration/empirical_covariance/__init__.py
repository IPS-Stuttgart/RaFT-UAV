"""Compatibility validation for empirical covariance sequence alignment.

The maintained implementation lives in the sibling ``empirical_covariance.py``
module. This package preserves the public import path while rejecting malformed
sequence identifiers before the legacy implementation stringifies them and can
silently align unrelated rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "empirical_covariance.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._empirical_covariance_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load empirical covariance implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ALIGNED_RESIDUALS = _IMPL.aligned_residuals


def _validate_sequence_ids(frame: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Reject missing, blank, or one-sided sequence identifiers."""

    frame_has_ids = "sequence_id" in frame.columns
    truth_has_ids = "sequence_id" in truth.columns
    if frame_has_ids != truth_has_ids:
        raise ValueError(
            "frame and truth must either both contain sequence_id or both omit it"
        )
    if not frame_has_ids:
        return

    for name, rows in (("frame", frame), ("truth", truth)):
        text = rows["sequence_id"].astype("string").str.strip()
        invalid = text.isna() | text.eq("").fillna(False)
        if not bool(invalid.any()):
            continue
        row_positions = np.flatnonzero(invalid.to_numpy(dtype=bool)).tolist()[:5]
        raise ValueError(
            f"{name} contains missing or blank sequence_id values: "
            f"sequence_id rows {row_positions}"
        )


def aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
) -> np.ndarray:
    """Align residuals only after validating sequence identity metadata."""

    _validate_sequence_ids(frame, truth)
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
globals()["_validate_sequence_ids"] = _validate_sequence_ids
globals()["aligned_residuals"] = aligned_residuals

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
