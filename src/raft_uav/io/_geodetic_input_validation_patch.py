"""Runtime validation for AERPAW geodetic coordinate inputs."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.coordinates import LocalENUProjector


_aerpaw = import_module("raft_uav.io.aerpaw")
_ORIGINAL_NORMALIZE_TRUTH = _aerpaw.normalize_truth
_ORIGINAL_NORMALIZE_RF = _aerpaw.normalize_rf
_ORIGINAL_NORMALIZE_RADAR = _aerpaw.normalize_radar
_ORIGINAL_PROJECTOR_FROM_TRUTH = _aerpaw.projector_from_truth


def _contains_complex(value: Any) -> bool:
    """Return whether a scalar or nested object array contains complex data."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(array):
        return True
    if array.dtype != object:
        return False
    return any(item is not value and _contains_complex(item) for item in array.flat)


def _reject_complex_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str,
) -> None:
    """Reject columns whose numeric representation is complex-valued."""

    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        if _contains_complex(values):
            raise ValueError(f"{context} {column} must contain real values")
        try:
            numeric = pd.to_numeric(values, errors="coerce")
        except (TypeError, ValueError):
            numeric = values
        if _contains_complex(numeric):
            raise ValueError(f"{context} {column} must contain real values")


def _origin_projector(values: Any) -> LocalENUProjector:
    """Build an ENU projector from exactly three validated scalar values."""

    error = "enu_origin_lla must contain latitude, longitude, and altitude"
    try:
        origin = tuple(values)
    except TypeError as exc:
        raise ValueError(error) from exc
    if len(origin) != 3:
        raise ValueError(error)
    return _projector_from_lla(origin[0], origin[1], origin[2])


def _projector_from_lla(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
) -> LocalENUProjector:
    """Create a projector without lossy real-valued pre-coercion."""

    return LocalENUProjector(
        origin_latitude_deg=latitude_deg,
        origin_longitude_deg=longitude_deg,
        origin_altitude_m=altitude_m,
    )


@wraps(_ORIGINAL_PROJECTOR_FROM_TRUTH)
def _projector_from_truth(truth: pd.DataFrame) -> LocalENUProjector:
    """Reject complex truth coordinates before selecting an ENU origin."""

    _reject_complex_columns(
        truth,
        ("latitude", "longitude", "altitude_m"),
        context="truth",
    )
    return _ORIGINAL_PROJECTOR_FROM_TRUTH(truth)


@wraps(_ORIGINAL_NORMALIZE_TRUTH)
def _normalize_truth(
    truth: pd.DataFrame,
    projector: LocalENUProjector | None = None,
    *,
    enu_origin_lla: Any = None,
) -> tuple[pd.DataFrame, LocalENUProjector, pd.Timestamp]:
    """Normalize truth without discarding complex coordinate components."""

    _reject_complex_columns(
        truth,
        ("latitude", "longitude", "altitude_m"),
        context="truth",
    )
    if enu_origin_lla is not None and projector is None:
        projector = _origin_projector(enu_origin_lla)
        enu_origin_lla = None
    return _ORIGINAL_NORMALIZE_TRUTH(
        truth,
        projector,
        enu_origin_lla=enu_origin_lla,
    )


@wraps(_ORIGINAL_NORMALIZE_RF)
def _normalize_rf(
    rf: pd.DataFrame,
    projector: LocalENUProjector,
    truth_origin_time: pd.Timestamp,
    default_std_m: float = 75.0,
    clock_offset_s: float = _aerpaw.DEFAULT_RF_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize RF rows after rejecting complex geodetic coordinates."""

    _reject_complex_columns(
        rf,
        ("Latitude", "Longitude"),
        context="RF",
    )
    return _ORIGINAL_NORMALIZE_RF(
        rf,
        projector,
        truth_origin_time,
        default_std_m=default_std_m,
        clock_offset_s=clock_offset_s,
    )


@wraps(_ORIGINAL_NORMALIZE_RADAR)
def _normalize_radar(
    radar: pd.DataFrame,
    projector: LocalENUProjector,
    truth_origin_time: pd.Timestamp,
    clock_offset_s: float = _aerpaw.DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> pd.DataFrame:
    """Normalize radar rows after rejecting complex geodetic coordinates."""

    _reject_complex_columns(
        radar,
        ("latitude", "longitude", "altitude_m"),
        context="radar",
    )
    return _ORIGINAL_NORMALIZE_RADAR(
        radar,
        projector,
        truth_origin_time,
        clock_offset_s=clock_offset_s,
    )


def install() -> None:
    """Install complex-geodetic validation on public and legacy entry points."""

    if getattr(_aerpaw, "_geodetic_input_validation_patch_applied", False):
        return

    targets = [_aerpaw]
    implementation = getattr(_aerpaw, "_IMPL", None)
    if implementation is not None:
        targets.append(implementation)

    for target in targets:
        target.projector_from_lla = _projector_from_lla
        target.projector_from_truth = _projector_from_truth
        target.normalize_truth = _normalize_truth
        target.normalize_rf = _normalize_rf
        target.normalize_radar = _normalize_radar

    _aerpaw._geodetic_input_validation_patch_applied = True
