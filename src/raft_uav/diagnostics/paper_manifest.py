"""Fail-closed manifest checks for paper-parity runs."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def validate_paper_manifest(
    *,
    file_manifest: Mapping[str, Any],
    enu_origin_mode: str,
    origin: Mapping[str, Any],
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
    require_explicit_origin: bool = True,
    require_file_hashes: bool = True,
) -> dict[str, Any]:
    """Validate the non-filter details that dominate paper reproducibility.

    The reference fingerprint is sensitive to file variant, timestamp reference,
    and ENU origin.  This helper raises on dangerous defaults and also returns a
    JSON-serializable report for manifests.
    """

    errors: list[str] = []
    warnings: list[str] = []
    if require_explicit_origin:
        if enu_origin_mode == "truth-first":
            errors.append("paper-parity runs must use an explicit site origin, not truth-first")
        _validate_origin(origin, errors=errors)
    for modality in ("rf", "radar", "truth"):
        entry = file_manifest.get(modality)
        if not isinstance(entry, Mapping):
            errors.append(f"missing {modality} file manifest")
            continue
        if not entry.get("path"):
            errors.append(f"missing {modality} file path")
        if not entry.get("variant"):
            warnings.append(f"missing {modality} file variant")
        if require_file_hashes and not entry.get("sha256"):
            warnings.append(f"missing {modality} sha256 digest")
    if not np.isfinite([rf_clock_offset_s, radar_clock_offset_s]).all():
        errors.append("RF/radar clock offsets must be finite")
    report = {
        "enabled": True,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "enu_origin_mode": enu_origin_mode,
        "rf_clock_offset_s": float(rf_clock_offset_s),
        "radar_clock_offset_s": float(radar_clock_offset_s),
    }
    if errors:
        raise ValueError("paper manifest validation failed: " + "; ".join(errors))
    return report


def _validate_origin(origin: Mapping[str, Any], *, errors: list[str]) -> None:
    lat = _optional_float(origin.get("latitude_deg"))
    lon = _optional_float(origin.get("longitude_deg"))
    alt = _optional_float(origin.get("altitude_m"))
    if lat is None or lon is None or alt is None:
        errors.append("explicit origin must include latitude_deg, longitude_deg, altitude_m")
        return
    if not -90.0 <= lat <= 90.0:
        errors.append(f"origin latitude outside [-90,90]: {lat}")
    if not -180.0 <= lon <= 180.0:
        errors.append(f"origin longitude outside [-180,180]: {lon}")
    if np.allclose([lat, lon, alt], [0.0, 0.0, 0.0], rtol=0.0, atol=1.0e-12):
        errors.append("origin must not be the placeholder 0,0,0")


def _optional_float(value: Any) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None
