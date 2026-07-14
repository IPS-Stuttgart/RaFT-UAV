"""Compatibility wrapper validating spread-guard estimate labels.

The maintained implementation lives in the sibling
``track5_estimate_ensemble_spread_guard.py`` module. This package preserves the
public import path while rejecting ambiguous normalized labels and unknown named
fallbacks instead of silently selecting a different estimate.
"""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_estimate_ensemble_spread_guard.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_estimate_ensemble_spread_guard_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load spread-guard implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_BUILD = _IMPL.build_spread_guarded_estimate_ensemble


def _materialize_unique_inputs(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
) -> tuple[tuple[tuple[str, pd.DataFrame, float], ...], tuple[str, ...]]:
    """Materialize inputs and reject labels that become ambiguous after normalization."""

    loaded_inputs = tuple(estimate_inputs)
    normalized_labels = tuple(_IMPL._safe_label(label) for label, _, _ in loaded_inputs)
    duplicates = sorted(
        label for label, count in Counter(normalized_labels).items() if count > 1
    )
    if duplicates:
        duplicate_text = ", ".join(repr(label) for label in duplicates)
        raise ValueError(
            "estimate input labels must be unique after normalization; "
            f"duplicates: {duplicate_text}"
        )
    return loaded_inputs, normalized_labels


def build_spread_guarded_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    spread_threshold_m: float,
    fallback_policy: str = "max-weight",
    fallback_label: str | None = None,
    fallback_blend: float = 0.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an ensemble after validating the label-based configuration."""

    loaded_inputs, normalized_labels = _materialize_unique_inputs(estimate_inputs)
    if loaded_inputs and fallback_policy == "label" and fallback_label:
        normalized_fallback = _IMPL._safe_label(fallback_label)
        if normalized_fallback not in normalized_labels:
            available = ", ".join(repr(label) for label in normalized_labels)
            raise ValueError(
                f"fallback_label {normalized_fallback!r} does not match any estimate "
                f"input label; available labels: {available}"
            )
    return _ORIGINAL_BUILD(
        loaded_inputs,
        template,
        spread_threshold_m=spread_threshold_m,
        fallback_policy=fallback_policy,
        fallback_label=fallback_label,
        fallback_blend=fallback_blend,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )


_IMPL.build_spread_guarded_estimate_ensemble = build_spread_guarded_estimate_ensemble

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_materialize_unique_inputs"] = _materialize_unique_inputs
globals()["build_spread_guarded_estimate_ensemble"] = (
    build_spread_guarded_estimate_ensemble
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
