"""Compatibility wrapper validating Track 5 speed-limit inputs.

The maintained implementation lives in the sibling ``track5_speed_limit.py``
module. This package preserves the public import path while rejecting malformed
iteration counts and invalid fixed-grid rows instead of silently coercing or
dropping them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_speed_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_speed_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load speed-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_PROJECT = _IMPL.project_track5_speed_limit
_NUMERIC_COLUMNS = (
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
)


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy or Boolean coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _validate_numeric_rows(submission: object) -> None:
    """Reject numeric rows that the legacy normalizer would silently discard."""

    rows = _IMPL._strip_csv_headers(pd.DataFrame(submission).copy())
    if any(column not in rows.columns for column in _NUMERIC_COLUMNS):
        return

    invalid: list[str] = []
    for column in _NUMERIC_COLUMNS:
        numeric = pd.to_numeric(rows[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if finite.all():
            continue
        row_positions = np.flatnonzero(~finite).tolist()
        invalid.append(f"{column} rows {row_positions}")
    if invalid:
        details = "; ".join(invalid)
        raise ValueError(f"submission contains non-finite numeric values: {details}")


def project_track5_speed_limit(
    submission: pd.DataFrame,
    *,
    max_speed_mps: float = 60.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project a trajectory after validating iteration and fixed-grid inputs."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    _validate_numeric_rows(submission)
    return _ORIGINAL_PROJECT(
        submission,
        max_speed_mps=max_speed_mps,
        iterations=validated_iterations,
        anchor_blend=anchor_blend,
    )


_IMPL.project_track5_speed_limit = project_track5_speed_limit

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_NUMERIC_COLUMNS"] = _NUMERIC_COLUMNS
globals()["_positive_integer"] = _positive_integer
globals()["_validate_numeric_rows"] = _validate_numeric_rows
globals()["project_track5_speed_limit"] = project_track5_speed_limit

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
