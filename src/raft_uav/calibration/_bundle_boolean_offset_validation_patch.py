"""Reject Boolean calibration-bundle time offsets at public boundaries."""

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


def _reject_boolean_offset(value: object, *, field_name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a real scalar, not a Boolean")


@dataclass(frozen=True)
class CalibrationBundle(_ORIGINAL_CALIBRATION_BUNDLE):
    """Calibration bundle with strict Boolean offset rejection."""

    def __post_init__(self) -> None:
        _reject_boolean_offset(
            self.rf_time_offset_s,
            field_name="rf_time_offset_s",
        )
        _reject_boolean_offset(
            self.radar_time_offset_s,
            field_name="radar_time_offset_s",
        )


def _manifest_offsets(payload: Mapping[str, Any]) -> tuple[object, object]:
    time_offsets = dict(payload.get("time_offsets") or {})
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
    """Load a bundle after rejecting Boolean timestamp corrections."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rf_offset, radar_offset = _manifest_offsets(payload)
        _reject_boolean_offset(rf_offset, field_name="rf_time_offset_s")
        _reject_boolean_offset(radar_offset, field_name="radar_time_offset_s")
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
    """Write a bundle manifest after rejecting Boolean offsets."""

    _reject_boolean_offset(rf_time_offset_s, field_name="rf_time_offset_s")
    _reject_boolean_offset(radar_time_offset_s, field_name="radar_time_offset_s")
    _ORIGINAL_WRITE_CALIBRATION_BUNDLE_MANIFEST(
        path,
        rf_time_offset_s=rf_time_offset_s,
        radar_time_offset_s=radar_time_offset_s,
        bias_model_path=bias_model_path,
        uncertainty_model_path=uncertainty_model_path,
        metadata=metadata,
    )


CalibrationBundle.__module__ = _IMPL.__name__
CalibrationBundle.__qualname__ = "CalibrationBundle"
_IMPL.CalibrationBundle = CalibrationBundle
_IMPL.load_calibration_bundle = load_calibration_bundle
_IMPL.write_calibration_bundle_manifest = write_calibration_bundle_manifest
