"""Compatibility guard for nonphysical MMUAD radar polar ranges.

The maintained implementation lives in the sibling ``radar.py`` module. This
package preserves the public import path while ensuring that negative radar
ranges are discarded before they can be reflected through the sensor origin
into plausible-looking Cartesian detections.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

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
    """Convert radar rows after removing negative physical ranges."""

    return _ORIGINAL_RADAR_POLAR_FRAME_TO_CANDIDATES(
        _drop_negative_radar_ranges(frame),
        source=source,
        sequence_id=sequence_id,
        default_sequence_id=default_sequence_id,
        azimuth_convention=azimuth_convention,
        angle_unit=angle_unit,
        range_std_m=range_std_m,
        angle_std_deg=angle_std_deg,
        z_std_m=z_std_m,
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
globals()["radar_polar_frame_to_candidates"] = radar_polar_frame_to_candidates

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
