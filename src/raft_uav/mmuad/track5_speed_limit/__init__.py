"""Compatibility wrapper validating Track 5 speed-limit inputs.

The maintained implementation lives in the sibling ``track5_speed_limit.py``
module. This package preserves the public import path while rejecting malformed
iteration counts, Boolean pseudo-numbers, invalid fixed-grid rows, and duplicate
sequence/timestamp keys instead of silently coercing or dropping them.
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


def _reject_boolean_scalar(value: object, *, message: str) -> object:
    """Reject Python and NumPy Boolean pseudo-numbers."""

    scalar = value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        scalar = value.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    return value


def _validated_submission_rows(submission: object) -> pd.DataFrame:
    """Return stripped submission rows for compatibility-boundary validation."""

    return _IMPL._strip_csv_headers(pd.DataFrame(submission).copy()).reset_index(drop=True)


def _validate_numeric_rows(submission: object) -> None:
    """Reject numeric rows that the legacy normalizer would silently coerce or drop."""

    rows = _validated_submission_rows(submission)
    if any(column not in rows.columns for column in _NUMERIC_COLUMNS):
        return

    boolean_invalid: list[str] = []
    nonfinite_invalid: list[str] = []
    for column in _NUMERIC_COLUMNS:
        boolean = rows[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).to_numpy(dtype=bool)
        if boolean.any():
            row_positions = np.flatnonzero(boolean).tolist()
            boolean_invalid.append(f"{column} rows {row_positions}")

        numeric = pd.to_numeric(rows[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if finite.all():
            continue
        row_positions = np.flatnonzero(~finite).tolist()
        nonfinite_invalid.append(f"{column} rows {row_positions}")
    if boolean_invalid:
        details = "; ".join(boolean_invalid)
        raise ValueError(f"submission contains Boolean numeric values: {details}")
    if nonfinite_invalid:
        details = "; ".join(nonfinite_invalid)
        raise ValueError(f"submission contains non-finite numeric values: {details}")


def _validate_unique_sequence_times(submission: object) -> None:
    """Reject duplicate normalized sequence/timestamp keys within one trajectory."""

    rows = _validated_submission_rows(submission)
    required = {"sequence_id", "time_s"}
    if not required <= set(rows.columns):
        return

    numeric_time = pd.to_numeric(rows["time_s"], errors="coerce")
    finite_time = np.isfinite(numeric_time.to_numpy(dtype=float))
    if not finite_time.all():
        return

    keys = pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "time_s": numeric_time.to_numpy(dtype=float),
            "row_position": np.arange(len(rows), dtype=int),
        }
    )
    duplicate = keys.duplicated(subset=["sequence_id", "time_s"], keep=False)
    if not duplicate.any():
        return

    details: list[str] = []
    duplicate_keys = keys.loc[duplicate]
    for (sequence_id, time_s), group in duplicate_keys.groupby(
        ["sequence_id", "time_s"],
        sort=True,
    ):
        positions = group["row_position"].astype(int).tolist()
        details.append(
            f"sequence_id={sequence_id!r}, time_s={float(time_s)!r}, rows={positions}"
        )
    raise ValueError(
        "submission contains duplicate sequence/time rows: " + "; ".join(details)
    )


def project_track5_speed_limit(
    submission: pd.DataFrame,
    *,
    max_speed_mps: float = 60.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project a trajectory after validating scalar controls and fixed-grid inputs."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    validated_max_speed_mps = _reject_boolean_scalar(
        max_speed_mps,
        message="max_speed_mps must be positive and finite",
    )
    validated_anchor_blend = _reject_boolean_scalar(
        anchor_blend,
        message="anchor_blend must be finite and in [0, 1)",
    )
    _validate_numeric_rows(submission)
    _validate_unique_sequence_times(submission)
    return _ORIGINAL_PROJECT(
        submission,
        max_speed_mps=validated_max_speed_mps,
        iterations=validated_iterations,
        anchor_blend=validated_anchor_blend,
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
globals()["_reject_boolean_scalar"] = _reject_boolean_scalar
globals()["_validated_submission_rows"] = _validated_submission_rows
globals()["_validate_numeric_rows"] = _validate_numeric_rows
globals()["_validate_unique_sequence_times"] = _validate_unique_sequence_times
globals()["project_track5_speed_limit"] = project_track5_speed_limit

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
