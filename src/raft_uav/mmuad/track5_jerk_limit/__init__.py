"""Compatibility wrapper validating Track 5 jerk-limit controls.

The maintained implementation lives in the sibling ``track5_jerk_limit.py``
module. This package keeps the public import path while ensuring that jerk
values remain attached to the actual four rows of every valid finite-difference
window, malformed controls and grid rows cannot be silently coerced or dropped,
no-op repairs are not reported as applied trajectory changes, and multi-step
diagnostics report the net displacement of the final trajectory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    parse_official_classification_cell,
    parse_official_sequence_cell,
)

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_jerk_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_jerk_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 jerk-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_jerk_kinks
_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence
_ORIGINAL_REPAIR_SEQUENCE_ONCE = _IMPL._repair_sequence_once


def _finite_scalar(value: object, *, message: str) -> float:
    """Return a finite non-Boolean scalar float."""

    if np.ma.is_masked(value):
        raise ValueError(message)
    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        scalar = value.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _normalize_iterations(value: Any) -> int:
    """Return a positive integer iteration count or raise a stable error."""

    message = "iterations must be a positive finite integer"
    numeric = _finite_scalar(value, message=message)
    if numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _validated_step_controls(
    *,
    max_jerk_mps3: object,
    smoothness_weight: object,
    min_correction_m: object,
    max_correction_m: object,
    repair_blend: object,
) -> dict[str, float | None]:
    """Normalize and validate controls used by one jerk-repair step."""

    max_jerk = _finite_scalar(
        max_jerk_mps3,
        message="max_jerk_mps3 must be positive and finite",
    )
    if max_jerk <= 0.0:
        raise ValueError("max_jerk_mps3 must be positive and finite")

    smoothness = _finite_scalar(
        smoothness_weight,
        message="smoothness_weight must be finite and non-negative",
    )
    if smoothness < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")

    min_correction = _finite_scalar(
        min_correction_m,
        message="min_correction_m must be finite and non-negative",
    )
    if min_correction < 0.0:
        raise ValueError("min_correction_m must be finite and non-negative")

    max_correction = None
    if max_correction_m is not None:
        max_correction = _finite_scalar(
            max_correction_m,
            message="max_correction_m must be positive and finite",
        )
        if max_correction <= 0.0:
            raise ValueError("max_correction_m must be positive and finite")

    blend = _finite_scalar(
        repair_blend,
        message="repair_blend must be finite and in [0, 1]",
    )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("repair_blend must be finite and in [0, 1]")

    return {
        "max_jerk_mps3": max_jerk,
        "smoothness_weight": smoothness,
        "min_correction_m": min_correction,
        "max_correction_m": max_correction,
        "repair_blend": blend,
    }


def _validated_controls(
    *,
    max_jerk_mps3: object,
    smoothness_weight: object,
    min_correction_m: object,
    max_correction_m: object,
    iterations: object,
    repair_blend: object,
) -> dict[str, float | int | None]:
    """Normalize and validate every public jerk-repair control."""

    controls: dict[str, float | int | None] = _validated_step_controls(
        max_jerk_mps3=max_jerk_mps3,
        smoothness_weight=smoothness_weight,
        min_correction_m=min_correction_m,
        max_correction_m=max_correction_m,
        repair_blend=repair_blend,
    )
    controls["iterations"] = _normalize_iterations(iterations)
    return controls


def _row_index_preview(indices: list[Any]) -> str:
    """Render a bounded list of offending DataFrame row labels."""

    preview = ", ".join(str(index) for index in indices[:5])
    return f"{preview}, ..." if len(indices) > 5 else preview


def _normalized_submission_rejecting_invalid_rows(
    submission: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a submission without silently deleting malformed grid rows."""

    rows = pd.DataFrame(submission).copy()
    lower = {str(column).strip().lower() for column in rows.columns}
    if {"sequence", "timestamp", "position", "classification"}.issubset(lower):
        official = _IMPL.normalize_official_track5_results_frame(rows)
        positions = [
            _IMPL.parse_official_position_cell(value)
            for value in official["Position"]
        ]
        xyz = pd.DataFrame(
            positions,
            columns=["state_x_m", "state_y_m", "state_z_m"],
            index=official.index,
        )
        rows = pd.DataFrame(
            {
                "sequence_id": official["Sequence"],
                "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
                "state_x_m": xyz["state_x_m"],
                "state_y_m": xyz["state_y_m"],
                "state_z_m": xyz["state_z_m"],
                "Classification": official["Classification"],
            }
        )

    required = {
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "Classification",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"submission missing normalized columns: {sorted(missing)}")

    sequence_ids: list[str] = []
    invalid_sequence_indices: list[Any] = []
    for index, value in rows["sequence_id"].items():
        try:
            sequence_ids.append(parse_official_sequence_cell(value))
        except ValueError:
            sequence_ids.append("")
            invalid_sequence_indices.append(index)
    if invalid_sequence_indices:
        raise ValueError(
            "submission contains invalid sequence identifiers at row indices: "
            f"{_row_index_preview(invalid_sequence_indices)}"
        )
    rows["sequence_id"] = sequence_ids

    classifications: list[int] = []
    invalid_classification_indices: list[Any] = []
    for index, value in rows["Classification"].items():
        try:
            classifications.append(parse_official_classification_cell(value))
        except ValueError:
            classifications.append(0)
            invalid_classification_indices.append(index)
    if invalid_classification_indices:
        raise ValueError(
            "submission contains invalid Classification values at row indices: "
            f"{_row_index_preview(invalid_classification_indices)}"
        )
    rows["Classification"] = classifications

    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    finite_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    finite = np.isfinite(rows[list(finite_columns)].to_numpy(float)).all(axis=1)
    if not finite.all():
        invalid_indices = rows.index[~finite].tolist()
        raise ValueError(
            "submission contains non-finite time or position values "
            f"at row indices: {_row_index_preview(invalid_indices)}"
        )
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def repair_track5_jerk_kinks(submission, **kwargs):
    """Validate every scalar control before running the legacy jerk repair."""

    controls = _validated_controls(
        max_jerk_mps3=kwargs.get("max_jerk_mps3", 80.0),
        smoothness_weight=kwargs.get("smoothness_weight", 10.0),
        min_correction_m=kwargs.get("min_correction_m", 1.0),
        max_correction_m=kwargs.get("max_correction_m"),
        iterations=kwargs.get("iterations", 1),
        repair_blend=kwargs.get("repair_blend", 1.0),
    )
    kwargs.update(controls)
    return _ORIGINAL_REPAIR(submission, **kwargs)


def _repair_sequence(group, **kwargs):
    """Validate a repair loop and report displacement from input to final output."""

    controls = _validated_controls(
        max_jerk_mps3=kwargs["max_jerk_mps3"],
        smoothness_weight=kwargs["smoothness_weight"],
        min_correction_m=kwargs["min_correction_m"],
        max_correction_m=kwargs["max_correction_m"],
        iterations=kwargs["iterations"],
        repair_blend=kwargs["repair_blend"],
    )
    kwargs.update(controls)
    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    original_positions = group[coordinate_columns].to_numpy(float)
    repaired, diagnostics = _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)
    final_positions = repaired[coordinate_columns].to_numpy(float)
    net_displacement = np.linalg.norm(final_positions - original_positions, axis=1)
    moved = np.isfinite(net_displacement) & (net_displacement > 0.0)
    diagnostics = diagnostics.copy()
    diagnostics["jerk_limit_displacement_m"] = np.where(
        moved,
        net_displacement,
        0.0,
    )
    diagnostics["jerk_limit_applied"] = (
        diagnostics["jerk_limit_applied"].to_numpy(bool) & moved
    )
    return repaired, diagnostics


def _repair_sequence_once(group, **kwargs):
    """Validate step controls and report an applied repair only when a row moved."""

    controls = _validated_step_controls(
        max_jerk_mps3=kwargs["max_jerk_mps3"],
        smoothness_weight=kwargs["smoothness_weight"],
        min_correction_m=kwargs["min_correction_m"],
        max_correction_m=kwargs["max_correction_m"],
        repair_blend=kwargs["repair_blend"],
    )
    kwargs.update(controls)
    repaired, diagnostics = _ORIGINAL_REPAIR_SEQUENCE_ONCE(group, **kwargs)
    diagnostics = diagnostics.copy()
    displacement = diagnostics["jerk_limit_displacement_m"].to_numpy(float)
    moved = np.isfinite(displacement) & (displacement > 0.0)
    diagnostics["jerk_limit_applied"] = (
        diagnostics["jerk_limit_applied"].to_numpy(bool) & moved
    )
    return repaired, diagnostics


def _row_jerk_proxy_with_window_support(
    times: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Map every valid jerk window to the rows in that window."""

    count = len(times)
    row_jerk = np.full(count, np.nan, dtype=float)
    d3 = _IMPL._third_derivative_matrix(times)
    if d3.size == 0:
        return row_jerk
    jerk_windows = d3 @ np.asarray(xyz, dtype=float)
    norms = np.linalg.norm(jerk_windows, axis=1)
    for coefficients, norm in zip(d3, norms, strict=True):
        for row_index in np.flatnonzero(coefficients):
            if np.isnan(row_jerk[row_index]) or norm > row_jerk[row_index]:
                row_jerk[row_index] = float(norm)
    return row_jerk


_IMPL._normalized_submission = _normalized_submission_rejecting_invalid_rows
_IMPL.repair_track5_jerk_kinks = repair_track5_jerk_kinks
_IMPL._repair_sequence = _repair_sequence
_IMPL._repair_sequence_once = _repair_sequence_once
_IMPL._row_jerk_proxy = _row_jerk_proxy_with_window_support

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalized_submission"] = _normalized_submission_rejecting_invalid_rows
globals()["repair_track5_jerk_kinks"] = repair_track5_jerk_kinks
globals()["_repair_sequence"] = _repair_sequence
globals()["_repair_sequence_once"] = _repair_sequence_once
globals()["_finite_scalar"] = _finite_scalar
globals()["_normalize_iterations"] = _normalize_iterations
globals()["_validated_step_controls"] = _validated_step_controls
globals()["_validated_controls"] = _validated_controls
globals()["_row_index_preview"] = _row_index_preview
globals()["_row_jerk_proxy_with_window_support"] = _row_jerk_proxy_with_window_support

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
