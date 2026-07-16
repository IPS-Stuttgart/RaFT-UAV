"""Compatibility package validating NIS calibration measurement dimensions."""

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
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load NIS covariance utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_NORMALIZED_DIAGNOSTICS_FRAME = _IMPL._normalized_diagnostics_frame


def _normalized_diagnostics_frame(
    frame: pd.DataFrame,
    *,
    accepted_only: bool,
) -> pd.DataFrame:
    """Normalize diagnostics without truncating malformed measurement dimensions."""

    if "measurement_dim" in frame.columns:
        _validate_measurement_dimensions(frame["measurement_dim"])
    return _LEGACY_NORMALIZED_DIAGNOSTICS_FRAME(
        frame,
        accepted_only=accepted_only,
    )


def _validate_measurement_dimensions(values: pd.Series) -> None:
    """Reject non-real, non-finite, and fractional dimensions before integer casting."""

    raw = pd.Series(values)
    boolean_mask = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    complex_mask = raw.map(lambda value: isinstance(value, (complex, np.complexfloating)))
    invalid_scalar_type = boolean_mask | complex_mask
    if invalid_scalar_type.any():
        row_index = int(np.flatnonzero(invalid_scalar_type.to_numpy())[0])
        bad_index = raw.index[row_index]
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "diagnostics measurement_dim values must be real integer dimensions; "
            f"got {bad_value!r} at index {bad_index!r}"
        )

    numbers = pd.to_numeric(raw, errors="coerce")
    numeric = numbers.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    nonfinite = numbers.notna().to_numpy() & ~finite
    fractional = finite & (numeric != np.rint(numeric))
    invalid = nonfinite | fractional
    if invalid.any():
        row_index = int(np.flatnonzero(invalid)[0])
        bad_index = raw.index[row_index]
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "diagnostics measurement_dim values must be finite integer dimensions; "
            f"got {bad_value!r} at index {bad_index!r}"
        )


_IMPL._normalized_diagnostics_frame = _normalized_diagnostics_frame

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_diagnostics_frame"] = _normalized_diagnostics_frame
globals()["_validate_measurement_dimensions"] = _validate_measurement_dimensions

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
