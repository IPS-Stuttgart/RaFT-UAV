"""Dataset IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from raft_uav.io import aerpaw as _aerpaw

_ORIGINAL_DISCOVER_ATTR = "_discover_flights_before_rf_preference"
_original_discover_flights = getattr(
    _aerpaw,
    _ORIGINAL_DISCOVER_ATTR,
    _aerpaw.discover_flights,
)
setattr(_aerpaw, _ORIGINAL_DISCOVER_ATTR, _original_discover_flights)


def _prefer_canonical_rf_export(
    flight: _aerpaw.FlightPaths,
    *,
    variant: str,
) -> _aerpaw.FlightPaths:
    """Prefer canonical AADM exports over unrelated CSV artifacts."""

    canonical = sorted(flight.root.glob("AADM*.csv"))
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


_aerpaw.discover_flights = _discover_flights
_aerpaw._IMPL.discover_flights = _discover_flights
