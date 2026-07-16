"""Compatibility package validating sequence-classifier fusion weights.

The maintained implementation lives in the sibling
``sequence_classifier_fusion.py`` module. This package preserves the public
import path while rejecting malformed image fusion weights before model
selection, prediction, or artifact writes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "sequence_classifier_fusion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._sequence_classifier_fusion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load sequence-classifier fusion implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

FusionModelSpec = _IMPL.FusionModelSpec
FusionSelectionResult = _IMPL.FusionSelectionResult
_LEGACY_FUSE_SEQUENCE_PROBABILITIES = _IMPL.fuse_sequence_probabilities
_LEGACY_SELECT_TRAIN_SAFE_FUSION = _IMPL.select_train_safe_fusion


def _validated_image_weight(value: Any, *, name: str = "image_weight") -> float:
    """Return a finite non-Boolean scalar fusion weight in the unit interval."""

    message = f"{name} must be a finite real scalar in [0, 1]"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        weight = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(message)
    return weight


def fuse_sequence_probabilities(
    image_probabilities: pd.DataFrame,
    nonimage_probabilities: pd.DataFrame,
    *,
    image_weight: float,
    eval_labels: dict[str, str] | None = None,
    class_source: str | None = None,
) -> pd.DataFrame:
    """Blend probabilities after validating the reported fusion weight."""

    return _LEGACY_FUSE_SEQUENCE_PROBABILITIES(
        image_probabilities,
        nonimage_probabilities,
        image_weight=_validated_image_weight(image_weight),
        eval_labels=eval_labels,
        class_source=class_source,
    )


def select_train_safe_fusion(
    *,
    image_train_features: pd.DataFrame,
    nonimage_train_features: pd.DataFrame,
    image_predict_features: pd.DataFrame,
    nonimage_predict_features: pd.DataFrame,
    train_labels: dict[str, str],
    eval_labels: dict[str, str] | None = None,
    model_specs: list[FusionModelSpec],
    image_weights: list[float],
    cv_folds: int = 5,
    cv_random_state: int = 20260627,
    output_dir: Path | None = None,
) -> FusionSelectionResult:
    """Select fusion settings after validating every candidate weight."""

    validated_weights = [
        _validated_image_weight(value, name=f"image_weights[{index}]")
        for index, value in enumerate(image_weights)
    ]
    return _LEGACY_SELECT_TRAIN_SAFE_FUSION(
        image_train_features=image_train_features,
        nonimage_train_features=nonimage_train_features,
        image_predict_features=image_predict_features,
        nonimage_predict_features=nonimage_predict_features,
        train_labels=train_labels,
        eval_labels=eval_labels,
        model_specs=model_specs,
        image_weights=validated_weights,
        cv_folds=cv_folds,
        cv_random_state=cv_random_state,
        output_dir=output_dir,
    )


_IMPL._validated_image_weight = _validated_image_weight
_IMPL.fuse_sequence_probabilities = fuse_sequence_probabilities
_IMPL.select_train_safe_fusion = select_train_safe_fusion

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_image_weight"] = _validated_image_weight
globals()["fuse_sequence_probabilities"] = fuse_sequence_probabilities
globals()["select_train_safe_fusion"] = select_train_safe_fusion

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
