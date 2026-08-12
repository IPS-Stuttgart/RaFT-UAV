"""Enforce the documented six-dimensional radar velocity mode."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Any

_PATCH_MARKER = "_raft_uav_radar_velocity_strict_mode_patch_applied"
_REQUIRED_VELOCITY_COLUMNS = (
    "velocity_east_mps",
    "velocity_north_mps",
    "velocity_down_mps",
)


def _validate_velocity_rows(module: Any, frame: Any) -> None:
    """Require a complete finite NED velocity triplet for every retained row."""

    missing = [column for column in _REQUIRED_VELOCITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "include_velocity=True requires all radar velocity components; "
            f"missing columns: {missing}"
        )
    for index, row in frame.iterrows():
        if module._radar_velocity_vector_enu(row) is None:
            raise ValueError(
                "include_velocity=True requires finite velocity_east_mps, "
                "velocity_north_mps, and velocity_down_mps for every radar row; "
                f"invalid row index: {index!r}"
            )


def install() -> None:
    """Patch ``radar_measurements_to_enu`` so explicit velocity mode is strict."""

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
