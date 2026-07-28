"""Dataset IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from raft_uav.io import aerpaw as _aerpaw

_CANONICAL_RF_EXPORT_NAMES = ("AADM.csv", "AADM_rerun.csv")
_ORIGINAL_FIND_RF_ROOT_ATTR = "_find_rf_root_before_nested_underscore_support"
_original_find_rf_sensor_and_radar_root = getattr(
    _aerpaw,
    _ORIGINAL_FIND_RF_ROOT_ATTR,
    _aerpaw.find_rf_sensor_and_radar_root,
)
setattr(
    _aerpaw,
    _ORIGINAL_FIND_RF_ROOT_ATTR,
    _original_find_rf_sensor_and_radar_root,
)
_ORIGINAL_DISCOVER_ATTR = "_discover_flights_before_rf_preference"
_original_discover_flights = getattr(
    _aerpaw,
    _ORIGINAL_DISCOVER_ATTR,
    _aerpaw.discover_flights,
)
setattr(_aerpaw, _ORIGINAL_DISCOVER_ATTR, _original_discover_flights)


def _find_rf_sensor_and_radar_root(dataset_root: Path) -> Path:
    """Find nested underscored RF roots accepted by the direct-path loader."""

    try:
        return _original_find_rf_sensor_and_radar_root(dataset_root)
    except FileNotFoundError:
        root = Path(dataset_root)
        for candidate in root.rglob("RF_Sensor_and_Radar"):
            if candidate.is_dir():
                return candidate
        raise


def _prefer_canonical_rf_export(
    flight: _aerpaw.FlightPaths,
    *,
    variant: str,
) -> _aerpaw.FlightPaths:
    """Prefer exact canonical AADM exports over unrelated CSV artifacts."""

    canonical = [
        candidate
        for candidate in (
            flight.root / name for name in _CANONICAL_RF_EXPORT_NAMES
        )
        if candidate.is_file()
    ]
    if not canonical:
        return flight
    rf_csv = _aerpaw._preferred_variant(canonical, variant=variant)
    return replace(
        flight,
        rf_csv=rf_csv,
        rf_variant=_aerpaw._path_variant(rf_csv),
    )


def _discover_flights(
    dataset_root: Path,
    *,
    variant: str = "auto",
) -> list[_aerpaw.FlightPaths]:
    """Discover flights with deterministic canonical RF-file preference."""

    flights = _original_discover_flights(dataset_root, variant=variant)
    return [
        _prefer_canonical_rf_export(flight, variant=variant)
        for flight in flights
    ]


_aerpaw.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw._IMPL.find_rf_sensor_and_radar_root = _find_rf_sensor_and_radar_root
_aerpaw.discover_flights = _discover_flights
_aerpaw._IMPL.discover_flights = _discover_flights
