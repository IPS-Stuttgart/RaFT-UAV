"""Compatibility wrapper for safe Track 5 Hampel repair.

The maintained implementation lives in the sibling ``track5_hampel_repair.py``
module. This package preserves the public import path while rejecting malformed
grid rows and invalid scalar controls before repair, and keeps sequence
endpoints fixed during local-median replacement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_hampel_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_hampel_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 Hampel-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_REPAIR = _IMPL.repair_track5_hampel_spikes
_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence


def _normalize_positive_integer(value: Any, *, field: str) -> int:
    """Return a positive integer scalar or raise a field-specific error."""

    message = f"{field} must be a positive finite integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _finite_scalar(value: object, *, message: str) -> float:
    """Return a finite non-Boolean scalar float or raise ``ValueError``."""

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


def _validated_float_controls(
    *,
    sigma_threshold: object,
    min_scale_m: object,
    min_residual_m: object,
    repair_blend: object,
) -> dict[str, float]:
    """Normalize and validate the scalar controls used by Hampel repair."""

    sigma = _finite_scalar(
        sigma_threshold,
        message="sigma_threshold must be positive and finite",
    )
    if sigma <= 0.0:
        raise ValueError("sigma_threshold must be positive and finite")

    min_scale = _finite_scalar(
        min_scale_m,
        message="min_scale_m must be positive and finite",
    )
    if min_scale <= 0.0:
        raise ValueError("min_scale_m must be positive and finite")

    min_residual = _finite_scalar(
        min_residual_m,
        message="min_residual_m must be finite and non-negative",
    )
    if min_residual < 0.0:
        raise ValueError("min_residual_m must be finite and non-negative")

    blend = _finite_scalar(
        repair_blend,
        message="repair_blend must be finite and in [0, 1]",
    )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("repair_blend must be finite and in [0, 1]")

    return {
        "sigma_threshold": sigma,
        "min_scale_m": min_scale,
        "min_residual_m": min_residual,
        "repair_blend": blend,
    }


def _load_track5_submission_frame_rejecting_invalid_rows(
    submission: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize a submission without silently deleting malformed grid rows."""

    rows = pd.DataFrame(submission).copy()
    normalized_columns = {
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
    }
    if normalized_columns.issubset(rows.columns):
        if "Classification" not in rows.columns:
            rows["Classification"] = rows.get("classification", 0)
        out = rows.copy()
    else:
        out = rows
        if not normalized_columns.issubset(out.columns):
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
            out = pd.DataFrame(
                {
                    "sequence_id": official["Sequence"].astype(str),
                    "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
                    "state_x_m": xyz["state_x_m"],
                    "state_y_m": xyz["state_y_m"],
                    "state_z_m": xyz["state_z_m"],
                    "Classification": official["Classification"],
                }
            )
    numeric_columns = (
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "Classification",
    )
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite_columns = ("time_s", "state_x_m", "state_y_m", "state_z_m")
    finite = np.isfinite(out[list(finite_columns)].to_numpy(float)).all(axis=1)
    if not finite.all():
        invalid_indices = out.index[~finite].tolist()
        preview = ", ".join(str(index) for index in invalid_indices[:5])
        suffix = ", ..." if len(invalid_indices) > 5 else ""
        raise ValueError(
            "submission contains non-finite time or position values "
            f"at row indices: {preview}{suffix}"
        )
    return out.copy().reset_index(drop=True)


def repair_track5_hampel_spikes(submission, **kwargs):
    """Validate every scalar control before running the legacy Hampel repair."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs.get("window_radius", 2),
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs.get("iterations", 1),
        field="iterations",
    )
    kwargs.update(
        _validated_float_controls(
            sigma_threshold=kwargs.get("sigma_threshold", 3.0),
            min_scale_m=kwargs.get("min_scale_m", 1.0),
            min_residual_m=kwargs.get("min_residual_m", 3.0),
            repair_blend=kwargs.get("repair_blend", 1.0),
        )
    )
    return _ORIGINAL_REPAIR(submission, **kwargs)


def _repair_sequence(group, **kwargs):
    """Validate direct calls to the private sequence repair loop."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs["window_radius"],
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs["iterations"],
        field="iterations",
    )
    kwargs.update(
        _validated_float_controls(
            sigma_threshold=kwargs["sigma_threshold"],
            min_scale_m=kwargs["min_scale_m"],
            min_residual_m=kwargs["min_residual_m"],
            repair_blend=kwargs["repair_blend"],
        )
    )
    return _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)


def _repair_xyz_once(
    work: pd.DataFrame,
    xyz: np.ndarray,
    original_xyz: np.ndarray,
    *,
    window_radius: int,
    sigma_threshold: float,
    min_scale_m: float,
    min_residual_m: float,
    repair_blend: float,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Repair eligible interior rows while leaving both endpoints unchanged."""

    window_radius = _normalize_positive_integer(
        window_radius,
        field="window_radius",
    )
    controls = _validated_float_controls(
        sigma_threshold=sigma_threshold,
        min_scale_m=min_scale_m,
        min_residual_m=min_residual_m,
        repair_blend=repair_blend,
    )
    sigma_threshold = controls["sigma_threshold"]
    min_scale_m = controls["min_scale_m"]
    min_residual_m = controls["min_residual_m"]
    repair_blend = controls["repair_blend"]

    out = xyz.copy()
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(xyz)):
        start = max(0, index - window_radius)
        stop = min(len(xyz), index + window_radius + 1)
        neighbor_indices = [item for item in range(start, stop) if item != index]
        neighbors = xyz[neighbor_indices]
        neighbors = neighbors[np.isfinite(neighbors).all(axis=1)]
        local_median = np.full(3, np.nan, dtype=float)
        robust_scale_m = np.nan
        residual_m = np.nan
        threshold_m = np.nan
        applied = False
        is_interior = 0 < index < len(xyz) - 1
        if is_interior and len(neighbors) >= 2 and np.isfinite(xyz[index]).all():
            local_median = np.median(neighbors, axis=0)
            neighbor_residuals = np.linalg.norm(neighbors - local_median, axis=1)
            robust_scale_m = max(
                float(np.median(neighbor_residuals) * 1.4826),
                min_scale_m,
            )
            residual_m = float(np.linalg.norm(xyz[index] - local_median))
            threshold_m = max(
                min_residual_m,
                sigma_threshold * robust_scale_m,
            )
            if residual_m > threshold_m:
                out[index] = (
                    (1.0 - repair_blend) * xyz[index]
                    + repair_blend * local_median
                )
                applied = True
        diagnostics.append(
            {
                "sequence_id": work.loc[index, "sequence_id"],
                "time_s": float(work.loc[index, "time_s"]),
                "iteration": int(iteration),
                "local_window_start_index": int(start),
                "local_window_stop_index": int(stop),
                "local_neighbor_count": int(len(neighbors)),
                "local_median_x_m": (
                    float(local_median[0]) if np.isfinite(local_median[0]) else np.nan
                ),
                "local_median_y_m": (
                    float(local_median[1]) if np.isfinite(local_median[1]) else np.nan
                ),
                "local_median_z_m": (
                    float(local_median[2]) if np.isfinite(local_median[2]) else np.nan
                ),
                "hampel_residual_m": residual_m,
                "hampel_scale_m": robust_scale_m,
                "hampel_threshold_m": threshold_m,
                "hampel_iteration_applied": bool(applied),
                "original_state_x_m": float(original_xyz[index, 0]),
                "original_state_y_m": float(original_xyz[index, 1]),
                "original_state_z_m": float(original_xyz[index, 2]),
            }
        )
    return out, pd.DataFrame.from_records(
        diagnostics,
        columns=_IMPL._diagnostic_columns(),
    )


_IMPL.load_track5_submission_frame = _load_track5_submission_frame_rejecting_invalid_rows
_IMPL.repair_track5_hampel_spikes = repair_track5_hampel_spikes
_IMPL._repair_sequence = _repair_sequence
_IMPL._repair_xyz_once = _repair_xyz_once

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["repair_track5_hampel_spikes"] = repair_track5_hampel_spikes
globals()["_repair_sequence"] = _repair_sequence
globals()["_repair_xyz_once"] = _repair_xyz_once
globals()["_normalize_positive_integer"] = _normalize_positive_integer
globals()["_finite_scalar"] = _finite_scalar
globals()["_validated_float_controls"] = _validated_float_controls

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
