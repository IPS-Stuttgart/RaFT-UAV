"""Compatibility guards for nonphysical MMUAD radar inputs.

The maintained implementation lives in the sibling ``radar.py`` module. This
package preserves the public import path while ensuring that negative radar
ranges are discarded, malformed uncertainty parameters are rejected, and
lossy Boolean or complex radar geometry cells cannot become plausible-looking
Cartesian detections.
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
_RADAR_GEOMETRY_COLUMNS = frozenset(
    {
        "range_m",
        "range",
        "r",
        "rho",
        "distance_m",
        "distance",
        "azimuth_rad",
        "az_rad",
        "bearing_rad",
        "azimuth_deg",
        "az_deg",
        "bearing_deg",
        "azimuth",
        "az",
        "bearing",
        "elevation_rad",
        "el_rad",
        "pitch_rad",
        "elevation_deg",
        "el_deg",
        "pitch_deg",
        "elevation",
        "el",
        "pitch",
    }
)


def _radar_geometry_cell(
    value: Any,
    *,
    column: object,
    row: object,
) -> Any:
    """Return one geometry cell after rejecting lossy pseudo-numbers."""

    seen: set[int] = set()
    current = value
    while isinstance(current, np.ndarray) and current.ndim == 0:
        identity = id(current)
        if identity in seen:
            raise ValueError(
                f"radar geometry column {column!r} contains a cyclic scalar "
                f"at row {row!r}"
            )
        seen.add(identity)
        current = current.item()

    if np.ma.is_masked(current):
        return np.nan
    if isinstance(current, (bool, np.bool_)):
        raise ValueError(
            f"radar geometry column {column!r} contains a Boolean value "
            f"at row {row!r}"
        )
    if np.iscomplexobj(current):
        raise ValueError(
            f"radar geometry column {column!r} contains a complex value "
            f"at row {row!r}"
        )
    return current


def _normalize_radar_geometry_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Inspect geometry cells before pandas or NumPy can coerce their types."""

    normalized = frame.copy()
    for column in normalized.columns:
        if str(column).strip().lower() not in _RADAR_GEOMETRY_COLUMNS:
            continue
        normalized[column] = pd.Series(
            (
                _radar_geometry_cell(value, column=column, row=row)
                for row, value in normalized[column].items()
            ),
            index=normalized.index,
        )
    return normalized


def _drop_negative_radar_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized radar frame without physically invalid ranges."""

    normalized = _normalize_radar_geometry_cells(frame)
    normalized = _IMPL.normalize_time_column_aliases(normalized, target="time_s")
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
globals()["_radar_geometry_cell"] = _radar_geometry_cell
globals()["_normalize_radar_geometry_cells"] = _normalize_radar_geometry_cells
globals()["_drop_negative_radar_ranges"] = _drop_negative_radar_ranges
globals()["_validated_radar_std"] = _validated_radar_std
globals()["radar_polar_frame_to_candidates"] = radar_polar_frame_to_candidates

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
