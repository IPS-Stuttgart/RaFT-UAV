"""Compatibility wrapper for soft class-conditioned Track 5 ensembling.

The maintained implementation lives in the sibling
``track5_soft_class_ensemble.py`` module. This package preserves the public
import path while canonicalizing exactly integer-equivalent classifier labels
such as ``0.0`` before the legacy implementation constructs one-hot class
probabilities and validating scalar ensemble controls before file access.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import (
    _validate_ensemble_weight,
    _validate_trim_fraction,
)

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_soft_class_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_soft_class_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load soft class ensemble implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_BUILD = _IMPL.build_soft_class_conditioned_estimate_ensemble


def _predicted_class_labels(values: pd.Series) -> pd.Series:
    """Return canonical official class-id strings from exact integer values."""

    raw = pd.Series(values)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    boolean_values = raw.map(lambda value: isinstance(value, bool | np.bool_)).to_numpy(bool)
    integer_like = (
        np.isfinite(numeric_array)
        & (numeric_array == np.rint(numeric_array))
        & ~boolean_values
    )
    if integer_like.any():
        positions = np.flatnonzero(integer_like)
        text.iloc[positions] = np.rint(numeric_array[positions]).astype(int).astype(str)
    return text


def _validate_predicted_class_labels(labels: pd.Series) -> None:
    """Reject non-empty classifier labels outside the official class IDs."""

    allowed_labels = tuple(_IMPL._official_class_labels())
    text = pd.Series(labels).fillna("").astype(str).str.strip()
    present = text.ne("")
    invalid = present & ~text.isin(allowed_labels)
    if not invalid.any():
        return
    examples = sorted(text.loc[invalid].unique())
    allowed = ", ".join(allowed_labels)
    raise ValueError(
        "predicted_class values must be official Track 5 class IDs "
        f"{{{allowed}}}; got {examples}"
    )


def _normalized_weight_map(raw: Any, inputs: tuple[Any, ...]) -> dict[str, float]:
    """Normalize one weight mapping after strict scalar validation."""

    if not isinstance(raw, dict):
        raise ValueError("weights must be an object")
    labels = [item.label for item in inputs]
    weights: dict[str, float] = {}
    for label in labels:
        weights[label] = _validate_ensemble_weight(
            raw.get(label, 0.0),
            label=label,
        )
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("weight sum must be positive")
    return {label: value / total for label, value in weights.items()}


def _validate_weight_config(
    weight_config: Any,
    inputs: tuple[Any, ...],
) -> None:
    """Validate global and class-specific weights before estimate-file access."""

    if not isinstance(weight_config, dict):
        raise ValueError("weight config must be an object")
    _normalized_weight_map(_IMPL._global_weights(weight_config), inputs)
    class_weights = weight_config.get("class_weights", {})
    if class_weights is None:
        class_weights = {}
    if not isinstance(class_weights, dict):
        raise ValueError("weight config class_weights must be an object")
    for raw_weights in class_weights.values():
        _normalized_weight_map(raw_weights, inputs)


def _select_trim_fraction(override: Any, weight_config: dict[str, Any]) -> float:
    """Select and strictly validate the configured trim fraction."""

    value = override if override is not None else weight_config.get("trim_fraction", 0.2)
    return _validate_trim_fraction(value)


def build_soft_class_conditioned_estimate_ensemble(
    estimate_inputs: Iterable[Any],
    *,
    template: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    weight_config: dict[str, Any],
    aggregation_policy: str | None = None,
    trim_fraction: Any = None,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a soft class ensemble after validating scalar controls up front."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    _validate_weight_config(weight_config, inputs)
    trim = _select_trim_fraction(trim_fraction, weight_config)
    return _LEGACY_BUILD(
        inputs,
        template=template,
        class_probabilities=class_probabilities,
        weight_config=weight_config,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


def _normalize_probability_rows(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Normalize soft probabilities and fill empty rows from hard class labels."""

    rows = pd.DataFrame(probabilities).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id"])
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("class probabilities must contain sequence_id/Sequence")
    out = pd.DataFrame({"sequence_id": rows[sequence_column].astype(str)})
    labels = tuple(_IMPL._official_class_labels())
    found_probability = False
    for label in labels:
        column = _IMPL._probability_column(rows, label)
        if column is not None:
            out[f"class_prob_{label}"] = pd.to_numeric(rows[column], errors="coerce")
            found_probability = True

    predicted_column = _IMPL._first_present(rows, _IMPL.PREDICTED_CLASS_ALIASES)
    if not found_probability:
        if predicted_column is None:
            raise ValueError("class probabilities need probability columns or predicted_class")
        predicted = _predicted_class_labels(rows[predicted_column])
        _validate_predicted_class_labels(predicted)
        for label in labels:
            out[f"class_prob_{label}"] = (predicted == label).astype(float)
    elif predicted_column is not None:
        predicted = _predicted_class_labels(rows[predicted_column])
        probability_columns = [f"class_prob_{label}" for label in labels]
        for column in probability_columns:
            if column not in out.columns:
                out[column] = 0.0
        usable_mass = out[probability_columns].apply(pd.to_numeric, errors="coerce")
        usable_mass = usable_mass.where(np.isfinite(usable_mass), 0.0).clip(lower=0.0)
        fallback = usable_mass.sum(axis=1).le(0.0) & predicted.ne("")
        _validate_predicted_class_labels(predicted.loc[fallback])
        for label in labels:
            out.loc[fallback, f"class_prob_{label}"] = (
                predicted.loc[fallback] == label
            ).astype(float)

    out = out.groupby("sequence_id", as_index=False).mean(numeric_only=True)
    return _IMPL._normalize_probability_mass(out)


_IMPL._normalized_weight_map = _normalized_weight_map
_IMPL._select_trim_fraction = _select_trim_fraction
_IMPL.build_soft_class_conditioned_estimate_ensemble = (
    build_soft_class_conditioned_estimate_ensemble
)
_IMPL._normalize_probability_rows = _normalize_probability_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

# Keep the patched helpers available for focused tests and exploratory use.
globals()["_predicted_class_labels"] = _predicted_class_labels
globals()["_validate_predicted_class_labels"] = _validate_predicted_class_labels
globals()["_normalized_weight_map"] = _normalized_weight_map
globals()["_validate_weight_config"] = _validate_weight_config
globals()["_select_trim_fraction"] = _select_trim_fraction
globals()["build_soft_class_conditioned_estimate_ensemble"] = (
    build_soft_class_conditioned_estimate_ensemble
)
globals()["_normalize_probability_rows"] = _normalize_probability_rows
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
