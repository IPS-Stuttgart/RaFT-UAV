"""LOFO-safe calibration bundle loading and application.

The bundle is a lightweight manifest that lets a tracking run apply all
calibration layers selected on training flights only:

* residual RF/radar time offsets,
* RF/radar bias-correction models, and
* learned heteroscedastic covariance models.

Paths in the manifest are resolved relative to the manifest location.  A minimal
manifest looks like::

    {
      "schema_version": 1,
      "time_offsets": {"rf": -0.25, "radar": 0.5},
      "bias_model_path": "bias_model.json",
      "uncertainty_model_path": "uncertainty_model.json"
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import (
    SensorBiasCorrectionModel,
    apply_bias_correction_models,
    bias_correction_summary,
    load_bias_correction_models,
)
from raft_uav.calibration.time_offset import apply_time_offset
from raft_uav.uncertainty import HeteroscedasticUncertaintyModel, load_uncertainty_model


@dataclass(frozen=True)
class CalibrationBundle:
    """Loaded calibration manifest and its model objects."""

    path: Path
    rf_time_offset_s: float = 0.0
    radar_time_offset_s: float = 0.0
    bias_models: Mapping[str, SensorBiasCorrectionModel] = field(default_factory=dict)
    uncertainty_model: HeteroscedasticUncertaintyModel | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe run summary for metrics artifacts."""

        uncertainty_summary: dict[str, Any] | None = None
        if self.uncertainty_model is not None:
            uncertainty_summary = {
                "model_type": "heteroscedastic-loglinear-variance",
                "metadata": dict(self.uncertainty_model.metadata),
                "heads": [
                    {
                        "source": head.source,
                        "dimension": head.dimension,
                        "training_rows": int(head.training_rows),
                        "min_std_m": float(head.min_std_m),
                        "max_std_m": float(head.max_std_m),
                    }
                    for head in self.uncertainty_model.heads
                ],
            }
        return {
            "path": str(self.path),
            "rf_time_offset_s": float(self.rf_time_offset_s),
            "radar_time_offset_s": float(self.radar_time_offset_s),
            "bias_models": bias_correction_summary(self.bias_models)
            if self.bias_models
            else {},
            "uncertainty_model": uncertainty_summary,
            "metadata": dict(self.metadata),
        }


def load_calibration_bundle(path: str | Path) -> CalibrationBundle:
    """Load a calibration bundle manifest and all referenced model files."""

    bundle_path = Path(path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError(f"unsupported calibration bundle schema {payload.get('schema_version')!r}")

    time_offsets = dict(payload.get("time_offsets") or {})
    rf_offset = _optional_float(
        payload.get(
            "rf_time_offset_correction_s",
            time_offsets.get("rf", time_offsets.get("rf_time_offset_s", 0.0)),
        )
    )
    radar_offset = _optional_float(
        payload.get(
            "radar_time_offset_correction_s",
            time_offsets.get("radar", time_offsets.get("radar_time_offset_s", 0.0)),
        )
    )

    bias_path = _optional_path(
        payload,
        bundle_path,
        "bias_model_path",
        "bias_models_path",
        "bias_path",
    )
    uncertainty_path = _optional_path(
        payload,
        bundle_path,
        "uncertainty_model_path",
        "heteroscedastic_model_path",
        "uncertainty_path",
    )
    return CalibrationBundle(
        path=bundle_path,
        rf_time_offset_s=0.0 if rf_offset is None else float(rf_offset),
        radar_time_offset_s=0.0 if radar_offset is None else float(radar_offset),
        bias_models={} if bias_path is None else load_bias_correction_models(bias_path),
        uncertainty_model=None if uncertainty_path is None else load_uncertainty_model(uncertainty_path),
        metadata=dict(payload.get("metadata") or {}),
    )


def apply_calibration_bundle(
    *,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    bundle: CalibrationBundle,
    apply_time_offsets: bool = True,
    apply_bias: bool = True,
    apply_uncertainty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the loaded calibration bundle to normalized RF/radar frames."""

    corrected_rf = rf.copy()
    corrected_radar = radar.copy()
    if apply_time_offsets:
        corrected_rf = _apply_offset_if_possible(corrected_rf, bundle.rf_time_offset_s)
        corrected_radar = _apply_offset_if_possible(corrected_radar, bundle.radar_time_offset_s)
    if apply_bias and bundle.bias_models:
        corrected_rf, corrected_radar = apply_bias_correction_models(
            rf=corrected_rf,
            radar=corrected_radar,
            models=bundle.bias_models,
        )
    if apply_uncertainty and bundle.uncertainty_model is not None:
        corrected_rf = _apply_uncertainty_if_available(
            corrected_rf,
            bundle.uncertainty_model,
            source="rf",
        )
        corrected_radar = _apply_uncertainty_if_available(
            corrected_radar,
            bundle.uncertainty_model,
            source="radar",
        )
    return corrected_rf, corrected_radar, bundle.summary()


def write_calibration_bundle_manifest(
    path: str | Path,
    *,
    rf_time_offset_s: float = 0.0,
    radar_time_offset_s: float = 0.0,
    bias_model_path: str | Path | None = None,
    uncertainty_model_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a calibration bundle manifest that references already-trained models."""

    destination = Path(path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "time_offsets": {
            "rf": float(rf_time_offset_s),
            "radar": float(radar_time_offset_s),
        },
        "metadata": dict(metadata or {}),
    }
    if bias_model_path is not None:
        payload["bias_model_path"] = _relative_or_string(destination, Path(bias_model_path))
    if uncertainty_model_path is not None:
        payload["uncertainty_model_path"] = _relative_or_string(
            destination,
            Path(uncertainty_model_path),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def _apply_offset_if_possible(frame: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns or float(offset_s) == 0.0:
        return frame
    return apply_time_offset(frame, offset_s)


def _apply_uncertainty_if_available(
    frame: pd.DataFrame,
    model: HeteroscedasticUncertaintyModel,
    *,
    source: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    try:
        return model.apply(frame, source=source)
    except ValueError as exc:
        if f"no heads for source {source!r}" in str(exc):
            return frame
        raise


def _optional_path(payload: Mapping[str, Any], bundle_path: Path, *keys: str) -> Path | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            path = Path(str(value))
            return path if path.is_absolute() else bundle_path.parent / path
    return None


def _optional_float(value: object) -> float | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _relative_or_string(manifest_path: Path, model_path: Path) -> str:
    try:
        return str(model_path.relative_to(manifest_path.parent))
    except ValueError:
        return str(model_path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


__all__ = [
    "CalibrationBundle",
    "apply_calibration_bundle",
    "load_calibration_bundle",
    "write_calibration_bundle_manifest",
]
