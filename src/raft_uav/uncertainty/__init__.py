from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "uncertainty.py"
_LEGACY_NAME = "_raft_uav_uncertainty_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _legacy
_SPEC.loader.exec_module(_legacy)

VarianceHead = _legacy.VarianceHead
HeteroscedasticUncertaintyModel = _legacy.HeteroscedasticUncertaintyModel
_original_variance_head_init = VarianceHead.__init__
_original_model_init = HeteroscedasticUncertaintyModel.__init__


def _finite_scalar(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _validate_variance_head(head) -> None:
    source = head.source
    if source not in _legacy.SOURCE_DIMS:
        raise ValueError(f"unknown uncertainty source {source!r}")
    if head.dimension not in _legacy.SOURCE_DIMS[source]:
        raise ValueError(
            f"dimension {head.dimension!r} is invalid for uncertainty source {source!r}"
        )

    feature_names = tuple(head.feature_names)
    allowed_features = (
        set(_legacy.RF_FEATURES) if source == "rf" else set(_legacy.RADAR_FEATURES)
    )
    if not feature_names:
        raise ValueError("variance head must contain at least one feature")
    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError("variance head feature names must be non-empty strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("variance head feature names must be unique")
    unknown_features = sorted(set(feature_names) - allowed_features)
    if unknown_features:
        raise ValueError(f"unknown {source} uncertainty features: {unknown_features}")

    try:
        coefficients = np.asarray(head.coefficients, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("variance head coefficients must be finite numbers") from exc
    if coefficients.size != len(feature_names):
        raise ValueError("feature/coefficient dimension mismatch")
    if not np.isfinite(coefficients).all():
        raise ValueError("variance head coefficients must be finite numbers")

    min_std_m = _finite_scalar(head.min_std_m, "min_std_m")
    max_std_m = _finite_scalar(head.max_std_m, "max_std_m")
    if min_std_m <= 0.0:
        raise ValueError("min_std_m must be positive")
    if max_std_m < min_std_m:
        raise ValueError("max_std_m must be greater than or equal to min_std_m")


def _validated_variance_head_init(
    self,
    source,
    dimension,
    feature_names,
    coefficients,
    min_std_m,
    max_std_m,
    training_rows,
):
    _original_variance_head_init(
        self,
        source,
        dimension,
        feature_names,
        coefficients,
        min_std_m,
        max_std_m,
        training_rows,
    )
    _validate_variance_head(self)


def _validated_model_init(self, heads, metadata):
    _original_model_init(self, heads, metadata)
    if not isinstance(self.metadata, Mapping):
        raise ValueError("uncertainty model metadata must be a mapping")
    seen: set[tuple[str, str]] = set()
    for head in self.heads:
        if not isinstance(head, VarianceHead):
            raise ValueError("uncertainty model heads must be VarianceHead instances")
        key = (head.source, head.dimension)
        if key in seen:
            raise ValueError(
                f"duplicate uncertainty variance head for {head.source!r}/{head.dimension!r}"
            )
        seen.add(key)


VarianceHead.__init__ = _validated_variance_head_init
HeteroscedasticUncertaintyModel.__init__ = _validated_model_init

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)
globals()["VarianceHead"] = VarianceHead
globals()["HeteroscedasticUncertaintyModel"] = HeteroscedasticUncertaintyModel
__doc__ = _legacy.__doc__
__all__ = [_name for _name in dir(_legacy) if not _name.startswith("__")]
