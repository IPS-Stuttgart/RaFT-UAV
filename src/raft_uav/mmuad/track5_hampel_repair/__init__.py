"""Compatibility wrapper for safe Track 5 Hampel repair.

The maintained implementation lives in the sibling ``track5_hampel_repair.py``
module. This package preserves the public import path while rejecting malformed
grid rows and invalid integer controls before repair.
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
_ORIGINAL_REPAIR_XYZ_ONCE = _IMPL._repair_xyz_once


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
    """Validate integer controls before running the legacy Hampel repair."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs.get("window_radius", 2),
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs.get("iterations", 1),
        field="iterations",
    )
    return _ORIGINAL_REPAIR(submission, **kwargs)


def _repair_sequence(group, **kwargs):
    """Validate private sequence-loop integer controls for direct callers."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs["window_radius"],
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs["iterations"],
        field="iterations",
    )
    return _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)


def _repair_xyz_once(work, xyz, original_xyz, **kwargs):
    """Validate the private local-window radius for direct callers."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs["window_radius"],
        field="window_radius",
    )
    return _ORIGINAL_REPAIR_XYZ_ONCE(work, xyz, original_xyz, **kwargs)


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

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
