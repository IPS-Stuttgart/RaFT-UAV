"""Compatibility package that rejects Boolean Track 5 ensemble weights.

The maintained implementation lives in the sibling ``track5_estimate_ensemble.py``
file. This wrapper preserves the public import surface while making weight
validation reject Boolean pseudo-numbers before empty-template returns, weight
configuration normalization, or estimate-file access.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy Track 5 estimate ensemble from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

EstimateInput = _IMPL.EstimateInput
_LEGACY_APPLY_WEIGHT_CONFIG = _IMPL.apply_estimate_weight_config
_LEGACY_BUILD = _IMPL.build_track5_estimate_ensemble
_LEGACY_WRITE = _IMPL.write_track5_estimate_ensemble_outputs


def _validate_ensemble_weight(weight: Any, *, label: str) -> float:
    """Return a finite non-negative weight, rejecting Boolean pseudo-numbers."""

    is_boolean_array = isinstance(weight, np.ndarray) and np.issubdtype(
        weight.dtype,
        np.bool_,
    )
    if isinstance(weight, (bool, np.bool_)) or is_boolean_array:
        raise ValueError(
            f"estimate weight must be finite and non-negative for {label}: {weight!r}"
        )
    try:
        parsed = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"estimate weight must be finite and non-negative for {label}: {weight!r}"
        ) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(
            f"estimate weight must be finite and non-negative for {label}: {parsed}"
        )
    return parsed


def _normalize_estimate_weight_mapping(raw_weights: dict[Any, Any]) -> dict[str, float]:
    safe_labels = _IMPL._normalize_unique_labels(
        raw_weights.keys(),
        context="ensemble weight",
    )
    weights: dict[str, float] = {}
    for raw_label, safe_label in zip(raw_weights.keys(), safe_labels):
        weights[safe_label] = _validate_ensemble_weight(
            raw_weights[raw_label],
            label=safe_label,
        )
    return weights


def _validated_runtime_inputs(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, Any]],
) -> list[tuple[str, pd.DataFrame, float]]:
    validated: list[tuple[str, pd.DataFrame, float]] = []
    for label, estimates, weight in estimate_inputs:
        safe_label = _IMPL._safe_label(str(label))
        validated.append(
            (
                label,
                estimates,
                _validate_ensemble_weight(weight, label=safe_label),
            )
        )
    return validated


def _validated_estimate_input_objects(
    estimate_inputs: Iterable[EstimateInput],
) -> list[EstimateInput]:
    validated: list[EstimateInput] = []
    for item in estimate_inputs:
        safe_label = _IMPL._safe_label(str(item.label))
        validated.append(
            EstimateInput(
                label=item.label,
                path=item.path,
                weight=_validate_ensemble_weight(item.weight, label=safe_label),
            )
        )
    return validated


def apply_estimate_weight_config(
    estimate_inputs: Iterable[EstimateInput],
    weights: dict[str, float],
    *,
    missing_policy: str = "error",
) -> list[EstimateInput]:
    """Apply a weight config without preserving Boolean inline weights."""

    updated = _LEGACY_APPLY_WEIGHT_CONFIG(
        estimate_inputs,
        weights,
        missing_policy=missing_policy,
    )
    return _validated_estimate_input_objects(updated)


def build_track5_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, Any]],
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an ensemble after validating all weights exactly once up front."""

    inputs = _validated_runtime_inputs(estimate_inputs)
    return _LEGACY_BUILD(
        inputs,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
    )


def write_track5_estimate_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
) -> dict[str, Path]:
    """Write ensemble outputs after validating weights before estimate-file I/O."""

    inputs = _validated_estimate_input_objects(estimate_inputs)
    return _LEGACY_WRITE(
        estimate_inputs=inputs,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
    )


_IMPL._validate_ensemble_weight = _validate_ensemble_weight
_IMPL._normalize_estimate_weight_mapping = _normalize_estimate_weight_mapping
_IMPL.apply_estimate_weight_config = apply_estimate_weight_config
_IMPL.build_track5_estimate_ensemble = build_track5_estimate_ensemble
_IMPL.write_track5_estimate_ensemble_outputs = write_track5_estimate_ensemble_outputs

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helpers available to direct imports and dependent wrappers.
globals()["_validate_ensemble_weight"] = _validate_ensemble_weight
globals()["_normalize_estimate_weight_mapping"] = _normalize_estimate_weight_mapping
globals()["_validated_runtime_inputs"] = _validated_runtime_inputs
globals()["_validated_estimate_input_objects"] = _validated_estimate_input_objects
globals()["apply_estimate_weight_config"] = apply_estimate_weight_config
globals()["build_track5_estimate_ensemble"] = build_track5_estimate_ensemble
globals()["write_track5_estimate_ensemble_outputs"] = (
    write_track5_estimate_ensemble_outputs
)
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
