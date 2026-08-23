"""Enforce complete radar velocity triplets without breaking 3-D fallback rows."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_raft_uav_radar_velocity_strict_mode_patch_applied"
_REQUIRED_VELOCITY_COLUMNS = (
    "velocity_east_mps",
    "velocity_north_mps",
    "velocity_down_mps",
)


def _has_velocity_value(value: Any) -> bool:
    """Return whether a component contains information rather than a missing marker."""

    if value is None or np.ma.is_masked(value):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    return True


def _validate_velocity_rows(module: Any, frame: Any) -> None:
    """Reject partial velocity data while preserving all-missing 3-D fallback rows.

    ``include_velocity=True`` is opportunistic for rows that carry no velocity
    information at all: those rows remain position-only measurements. Once any
    velocity component is present, however, the row must contain a complete
    finite NED triplet; silently dropping a partial triplet would hide corrupted
    or incompletely normalized radar data.
    """

    available_columns = {
        column for column in _REQUIRED_VELOCITY_COLUMNS if column in frame.columns
    }
    missing_columns = [
        column for column in _REQUIRED_VELOCITY_COLUMNS if column not in frame.columns
    ]

    for index, row in frame.iterrows():
        provided = {
            column: _has_velocity_value(row[column])
            for column in available_columns
        }
        if not any(provided.values()):
            # No velocity information was supplied for this row. Preserve the
            # long-standing position-only fallback used by normalized Fortem data.
            continue
        if missing_columns:
            raise ValueError(
                "include_velocity=True requires all radar velocity components "
                "when any component is present; "
                f"missing columns: {missing_columns}; invalid row index: {index!r}"
            )
        if not all(provided.get(column, False) for column in _REQUIRED_VELOCITY_COLUMNS):
            raise ValueError(
                "include_velocity=True requires complete finite radar velocity "
                f"components when any component is present; invalid row index: {index!r}"
            )
        if module._radar_velocity_vector_enu(row) is None:
            raise ValueError(
                "include_velocity=True requires finite velocity_east_mps, "
                "velocity_north_mps, and velocity_down_mps when velocity data is "
                f"present; invalid row index: {index!r}"
            )


def install() -> None:
    """Patch ``radar_measurements_to_enu`` with partial-triplet validation."""

    module = import_module("raft_uav.io.aerpaw")
    if getattr(module, _PATCH_MARKER, False):
        return

    original = module.radar_measurements_to_enu
    original_signature = signature(original)

    @wraps(original)
    def radar_measurements_to_enu(radar: Any, *args: Any, **kwargs: Any) -> Any:
        bound = original_signature.bind_partial(radar, *args, **kwargs)
        bound.apply_defaults()
        if not bool(bound.arguments["include_velocity"]):
            return original(*bound.args, **bound.kwargs)

        frame = radar
        if "east_m" not in frame.columns:
            projector = bound.arguments["projector"]
            truth_origin_time = bound.arguments["truth_origin_time"]
            if projector is None or truth_origin_time is None:
                return original(*bound.args, **bound.kwargs)
            frame = module.normalize_radar(
                frame,
                projector,
                truth_origin_time,
                clock_offset_s=bound.arguments["clock_offset_s"],
            )

        _validate_velocity_rows(module, frame)
        bound.arguments["radar"] = frame
        return original(*bound.args, **bound.kwargs)

    module.radar_measurements_to_enu = radar_measurements_to_enu
    setattr(module, _PATCH_MARKER, True)
