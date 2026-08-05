"""Validate calibration-bundle manifests, metadata, and time offsets at public boundaries."""

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


def _unwrapped_real_scalar(value: object, *, error: str) -> object:
    """Return a scalar payload without lossy nested-container coercion."""

    scalar = value
    seen_arrays: set[int] = set()
    while isinstance(scalar, (np.ndarray, np.generic)):
        if np.ma.is_masked(scalar):
            raise ValueError(error)
        if isinstance(scalar, np.ndarray):
            if scalar.ndim != 0:
                raise ValueError(error)
            marker = id(scalar)
            if marker in seen_arrays:
                raise ValueError(error)
            seen_arrays.add(marker)
        scalar = scalar.item()

    if (
        np.ma.is_masked(scalar)
        or isinstance(scalar, (bool, np.bool_))
        or isinstance(scalar, (complex, np.complexfloating))
    ):
        raise ValueError(error)
    try:
        scalar_array = np.asarray(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar_array.ndim != 0 or np.iscomplexobj(scalar_array):
        raise ValueError(error)
    return scalar


def _finite_real_offset(
    value: object,
    *,
    field_name: str,
    allow_nonfinite_missing: bool = False,
) -> float:
    error = f"{field_name} must be a finite real scalar"
    item = _unwrapped_real_scalar(value, error=error)
    try:
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) and not allow_nonfinite_missing:
        raise ValueError(error)
    return number


def _exact_schema_version(value: object) -> int:
    error = "schema_version must be an exact integer scalar"
    item = _unwrapped_real_scalar(value, error=error)
    try:
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(error)
    return int(number)


def _optional_mapping(
    value: object | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Return an optional manifest mapping without hiding malformed values."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping or null")
    return value


@dataclass(frozen=True)
class CalibrationBundle(_ORIGINAL_CALIBRATION_BUNDLE):
    """Calibration bundle with validated metadata and finite time offsets."""

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
        object.__setattr__(
            self,
            "metadata",
            dict(_optional_mapping(self.metadata, field_name="metadata")),
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
    """Load a bundle after validating its shape, schema, metadata, and offsets."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration bundle payload must be a mapping")
    schema_value = payload.get("schema_version", 1)
    schema_version = _exact_schema_version(schema_value)
    if schema_version != 1:
        raise ValueError(f"unsupported calibration bundle schema {schema_value!r}")
    _optional_mapping(payload.get("metadata"), field_name="metadata")
    rf_offset, radar_offset = _manifest_offsets(payload)
    if rf_offset is not None:
        _finite_real_offset(
            rf_offset,
            field_name="rf_time_offset_s",
            allow_nonfinite_missing=True,
        )
    if radar_offset is not None:
        _finite_real_offset(
            radar_offset,
            field_name="radar_time_offset_s",
            allow_nonfinite_missing=True,
        )
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
    """Write a bundle manifest after validating metadata and scalar offsets."""

    validated_rf_offset = _finite_real_offset(
        rf_time_offset_s,
        field_name="rf_time_offset_s",
        allow_nonfinite_missing=True,
    )
    validated_radar_offset = _finite_real_offset(
        radar_time_offset_s,
        field_name="radar_time_offset_s",
        allow_nonfinite_missing=True,
    )
    validated_metadata = _optional_mapping(metadata, field_name="metadata")
    _ORIGINAL_WRITE_CALIBRATION_BUNDLE_MANIFEST(
        path,
        rf_time_offset_s=validated_rf_offset,
        radar_time_offset_s=validated_radar_offset,
        bias_model_path=bias_model_path,
        uncertainty_model_path=uncertainty_model_path,
        metadata=validated_metadata,
    )


CalibrationBundle.__module__ = _IMPL.__name__
CalibrationBundle.__qualname__ = "CalibrationBundle"
_IMPL.CalibrationBundle = CalibrationBundle
_IMPL.load_calibration_bundle = load_calibration_bundle
_IMPL.write_calibration_bundle_manifest = write_calibration_bundle_manifest
