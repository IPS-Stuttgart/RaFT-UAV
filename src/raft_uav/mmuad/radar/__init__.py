"""Compatibility guards for nonphysical MMUAD radar inputs.

The maintained implementation lives in the sibling ``radar.py`` module. This
package preserves the public import path while ensuring that negative radar
ranges are discarded and malformed uncertainty parameters are rejected before
they can produce plausible-looking Cartesian detections with invalid metadata.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._radar_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD radar implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RADAR_POLAR_FRAME_TO_CANDIDATES = _IMPL.radar_polar_frame_to_candidates


def _drop_negative_radar_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized radar frame without physically invalid ranges."""

    normalized = _IMPL.normalize_time_column_aliases(frame, target="time_s")
    normalized = _IMPL._normalize_radar_columns(normalized)
    range_m = pd.to_numeric(normalized["range_m"], errors="coerce")
    negative = range_m.lt(0.0).fillna(False)
    if not bool(negative.any()):
        return normalized
    return normalized.loc[~negative].copy()


def _validated_radar_std(
    value: Any,
    *,
    name: str,
    allow_zero: bool,
) -> float:
    """Return one finite radar standard deviation or raise a precise error."""

    qualifier = "non-negative" if allow_zero else "positive"
    error = f"{name} must be a finite {qualifier} real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    if number < 0.0 or (not allow_zero and number == 0.0):
        raise ValueError(error)
    return number


def radar_polar_frame_to_candidates(
    frame: pd.DataFrame,
    *,
    source: str = "radar-polar",
    sequence_id: str | None = None,
    default_sequence_id: str = "default",
    azimuth_convention: str = "north-clockwise",
    angle_unit: str = "deg",
    range_std_m: float = 2.0,
    angle_std_deg: float = 2.0,
    z_std_m: float = 5.0,
) -> Any:
    """Convert radar rows after validating physical ranges and uncertainties."""

    validated_range_std_m = _validated_radar_std(
        range_std_m,
        name="range_std_m",
        allow_zero=False,
    )
    validated_angle_std_deg = _validated_radar_std(
        angle_std_deg,
        name="angle_std_deg",
        allow_zero=True,
    )
    validated_z_std_m = _validated_radar_std(
        z_std_m,
        name="z_std_m",
        allow_zero=False,
    )
    return _ORIGINAL_RADAR_POLAR_FRAME_TO_CANDIDATES(
        _drop_negative_radar_ranges(frame),
        source=source,
        sequence_id=sequence_id,
        default_sequence_id=default_sequence_id,
        azimuth_convention=azimuth_convention,
        angle_unit=angle_unit,
        range_std_m=validated_range_std_m,
        angle_std_deg=validated_angle_std_deg,
        z_std_m=validated_z_std_m,
    )


_IMPL.radar_polar_frame_to_candidates = radar_polar_frame_to_candidates

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_drop_negative_radar_ranges"] = _drop_negative_radar_ranges
globals()["_validated_radar_std"] = _validated_radar_std
globals()["radar_polar_frame_to_candidates"] = radar_polar_frame_to_candidates

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
