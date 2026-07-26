"""Validate calibration-bundle manifests and time offsets at public boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from raft_uav.calibration import bundle as _IMPL

_ORIGINAL_CALIBRATION_BUNDLE = _IMPL.CalibrationBundle
_ORIGINAL_LOAD_CALIBRATION_BUNDLE = _IMPL.load_calibration_bundle
_ORIGINAL_WRITE_CALIBRATION_BUNDLE_MANIFEST = _IMPL.write_calibration_bundle_manifest


def _finite_real_offset(value: object, *, field_name: str) -> float:
    error = f"{field_name} must be a finite real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number):
        raise ValueError(error)
    return number


def _exact_schema_version(value: object) -> int:
    error = "schema_version must be an exact integer scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(error)
    return int(number)


@dataclass(frozen=True)
class CalibrationBundle(_ORIGINAL_CALIBRATION_BUNDLE):
    """Calibration bundle with validated finite time offsets."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rf_time_offset_s",
            _finite_real_offset(self.rf_time_offset_s, field_name="rf_time_offset_s"),
        )
        object.__setattr__(
            self,
            "radar_time_offset_s",
            _finite_real_offset(
                self.radar_time_offset_s,
                field_name="radar_time_offset_s",
            ),
        )


def _manifest_offsets(payload: Mapping[str, Any]) -> tuple[object, object]:
    raw_time_offsets = payload.get("time_offsets")
    if raw_time_offsets is None:
        time_offsets: Mapping[str, Any] = {}
    elif isinstance(raw_time_offsets, Mapping):
        time_offsets = raw_time_offsets
    else:
        raise ValueError("time_offsets must be a mapping or null")
    rf_offset = payload.get(
        "rf_time_offset_correction_s",
        time_offsets.get("rf", time_offsets.get("rf_time_offset_s", 0.0)),
    )
    radar_offset = payload.get(
        "radar_time_offset_correction_s",
        time_offsets.get("radar", time_offsets.get("radar_time_offset_s", 0.0)),
    )
    return rf_offset, radar_offset


def load_calibration_bundle(path: str | Path) -> CalibrationBundle:
    """Load a bundle after validating its shape, schema, and time offsets."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration bundle payload must be a mapping")
    schema_value = payload.get("schema_version", 1)
    schema_version = _exact_schema_version(schema_value)
    if schema_version != 1:
        raise ValueError(f"unsupported calibration bundle schema {schema_value!r}")
    rf_offset, radar_offset = _manifest_offsets(payload)
    if rf_offset is not None:
        _finite_real_offset(rf_offset, field_name="rf_time_offset_s")
    if radar_offset is not None:
        _finite_real_offset(radar_offset, field_name="radar_time_offset_s")
    return _ORIGINAL_LOAD_CALIBRATION_BUNDLE(path)


def write_calibration_bundle_manifest(
    path: str | Path,
    *,
    rf_time_offset_s: float = 0.0,
    radar_time_offset_s: float = 0.0,
    bias_model_path: str | Path | None = None,
    uncertainty_model_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a bundle manifest after validating finite time offsets."""

    validated_rf_offset = _finite_real_offset(
        rf_time_offset_s,
        field_name="rf_time_offset_s",
    )
    validated_radar_offset = _finite_real_offset(
        radar_time_offset_s,
        field_name="radar_time_offset_s",
    )
    _ORIGINAL_WRITE_CALIBRATION_BUNDLE_MANIFEST(
        path,
        rf_time_offset_s=validated_rf_offset,
        radar_time_offset_s=validated_radar_offset,
        bias_model_path=bias_model_path,
        uncertainty_model_path=uncertainty_model_path,
        metadata=metadata,
    )


CalibrationBundle.__module__ = _IMPL.__name__
CalibrationBundle.__qualname__ = "CalibrationBundle"
_IMPL.CalibrationBundle = CalibrationBundle
_IMPL.load_calibration_bundle = load_calibration_bundle
_IMPL.write_calibration_bundle_manifest = write_calibration_bundle_manifest
