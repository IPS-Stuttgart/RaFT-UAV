"""Reject complex Track 5 repair cells before lossy float coercion."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_rejects_complex_repair_rows"
_NUMERIC_COLUMNS = (
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
)


def _is_complex_scalar(value: object) -> bool:
    """Return whether one scalar cell has a complex numeric dtype."""

    if np.ma.is_masked(value):
        return False
    try:
        scalar = np.asanyarray(value)
    except (TypeError, ValueError):
        return False
    if scalar.ndim != 0:
        return False
    if np.ma.isMaskedArray(scalar) and bool(np.ma.getmaskarray(scalar).any()):
        return False
    return scalar.dtype.kind == "c"


def _reject_complex_numeric_rows(submission: object) -> None:
    """Reject complex grid cells that pandas/NumPy would truncate to real values."""

    rows = pd.DataFrame(submission)
    invalid_details: list[str] = []
    for column in _NUMERIC_COLUMNS:
        if column not in rows.columns:
            continue
        invalid_positions = [
            position
            for position, value in enumerate(rows[column])
            if _is_complex_scalar(value)
        ]
        if invalid_positions:
            invalid_details.append(f"{column} rows {invalid_positions}")
    if invalid_details:
        raise ValueError(
            "submission contains complex numeric values: " + "; ".join(invalid_details)
        )


def _wrap_submission_validator(module: Any, function_name: str) -> None:
    """Install complex-cell validation around one repair input boundary."""

    original: Callable[..., Any] = getattr(module, function_name)
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated(submission: object, *args: Any, **kwargs: Any) -> Any:
        _reject_complex_numeric_rows(submission)
        return original(submission, *args, **kwargs)

    setattr(validated, _PATCH_MARKER, True)
    setattr(module, function_name, validated)
    implementation = getattr(module, "_IMPL", None)
    if implementation is not None and hasattr(implementation, function_name):
        setattr(implementation, function_name, validated)


def install() -> None:
    """Install complex-row guards on all Track 5 trajectory-repair inputs."""

    from raft_uav.mmuad import track5_acceleration_limit
    from raft_uav.mmuad import track5_hampel_repair
    from raft_uav.mmuad import track5_jerk_limit
    from raft_uav.mmuad import track5_vertical_repair

    _wrap_submission_validator(track5_acceleration_limit, "_validate_numeric_rows")
    _wrap_submission_validator(track5_acceleration_limit, "_normalized_submission")
    _wrap_submission_validator(track5_jerk_limit, "_normalized_submission")
    _wrap_submission_validator(track5_hampel_repair, "load_track5_submission_frame")
    _wrap_submission_validator(track5_vertical_repair, "_validate_numeric_rows")
