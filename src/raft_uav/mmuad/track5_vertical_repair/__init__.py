"""Compatibility wrapper validating vertical-repair inputs.

The maintained implementation lives in the sibling ``track5_vertical_repair.py``
module. This package preserves the public import path while rejecting malformed
``iterations`` values, invalid repair thresholds, and invalid numeric rows instead
of silently coercing or dropping them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_vertical_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_vertical_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load vertical-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_vertical_spikes
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


def _finite_nonnegative(value: object, *, name: str) -> float:
    """Return a finite non-negative threshold without Boolean coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _validate_numeric_rows(submission: object) -> None:
    """Reject numeric rows that the legacy normalizer would silently discard."""

    rows = pd.DataFrame(submission)
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


def repair_track5_vertical_spikes(
    submission,
    *,
    max_vertical_speed_mps: float = 20.0,
    max_neighbor_vertical_speed_mps: float = 10.0,
    max_vertical_residual_m: float = 15.0,
    max_horizontal_speed_mps: float | None = 80.0,
    iterations: int = 2,
):
    """Return vertically repaired estimates after validating public inputs."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    validated_vertical_speed = _finite_nonnegative(
        max_vertical_speed_mps,
        name="max_vertical_speed_mps",
    )
    validated_neighbor_vertical_speed = _finite_nonnegative(
        max_neighbor_vertical_speed_mps,
        name="max_neighbor_vertical_speed_mps",
    )
    validated_vertical_residual = _finite_nonnegative(
        max_vertical_residual_m,
        name="max_vertical_residual_m",
    )
    validated_horizontal_speed = (
        None
        if max_horizontal_speed_mps is None
        else _finite_nonnegative(
            max_horizontal_speed_mps,
            name="max_horizontal_speed_mps",
        )
    )
    _validate_numeric_rows(submission)
    return _ORIGINAL_REPAIR(
        submission,
        max_vertical_speed_mps=validated_vertical_speed,
        max_neighbor_vertical_speed_mps=validated_neighbor_vertical_speed,
        max_vertical_residual_m=validated_vertical_residual,
        max_horizontal_speed_mps=validated_horizontal_speed,
        iterations=validated_iterations,
    )


_IMPL.repair_track5_vertical_spikes = repair_track5_vertical_spikes

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_NUMERIC_COLUMNS"] = _NUMERIC_COLUMNS
globals()["_positive_integer"] = _positive_integer
globals()["_finite_nonnegative"] = _finite_nonnegative
globals()["_validate_numeric_rows"] = _validate_numeric_rows
globals()["repair_track5_vertical_spikes"] = repair_track5_vertical_spikes

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
