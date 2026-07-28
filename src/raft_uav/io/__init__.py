"""Dataset IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from raft_uav.io import aerpaw as _aerpaw

_CANONICAL_RF_EXPORT_NAMES = ("AADM.csv", "AADM_rerun.csv")
_ORIGINAL_DISCOVER_ATTR = "_discover_flights_before_rf_preference"
_original_discover_flights = getattr(
    _aerpaw,
    _ORIGINAL_DISCOVER_ATTR,
    _aerpaw.discover_flights,
)
setattr(_aerpaw, _ORIGINAL_DISCOVER_ATTR, _original_discover_flights)


def _prefer_canonical_rf_export(
    flight: _aerpaw.FlightPaths,
) -> _aerpaw.FlightPaths:
    """Prefer exact canonical AADM exports without crossing file variants."""

    canonical = [
        candidate
        for candidate in (
            flight.root / name for name in _CANONICAL_RF_EXPORT_NAMES
        )
        if candidate.is_file()
    ]
    if not canonical or flight.rf_variant not in {"original", "rerun"}:
        return flight
    rf_csv = _aerpaw._preferred_variant(canonical, variant=flight.rf_variant)
    if rf_csv is None:
        return flight
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
    return [_prefer_canonical_rf_export(flight) for flight in flights]


_aerpaw.discover_flights = _discover_flights
_aerpaw._IMPL.discover_flights = _discover_flights
