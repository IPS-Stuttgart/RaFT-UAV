"""Compatibility fixes for fifth-wave reliability diagnostics.

The maintained implementation lives in the sibling ``fifth_wave_diagnostics.py``
module. This package preserves the public import path while keeping serialized
track identifiers exact and validating bootstrap controls before empty-input
returns or lossy integer coercion.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "fifth_wave_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._fifth_wave_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load fifth-wave diagnostics from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

MetricFunction = _IMPL.MetricFunction
BootstrapInterval = _IMPL.BootstrapInterval
_ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL = _IMPL.block_bootstrap_interval
_ORIGINAL_PAIRED_DELTA_SUMMARY = _IMPL.paired_delta_summary


def _positive_integer_scalar(value: object, *, name: str) -> int:
    """Return a positive exact integer scalar."""

    normalized = optional_int(value)
    if normalized is None or normalized < 1:
        raise ValueError(f"{name} must be a positive integer scalar")
    return normalized


def _confidence_scalar(value: object) -> float:
    """Return a finite scalar confidence strictly between zero and one."""

    normalized = optional_float(value)
    if normalized is None or not 0.0 < normalized < 1.0:
        raise ValueError("confidence must be a finite real scalar in (0, 1)")
    return normalized


def block_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    metric: str | MetricFunction = "mean",
    block_size: int = 50,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapInterval:
    """Return a bootstrap interval after validating every public control."""

    _IMPL._metric_function(metric)
    validated_block_size = _positive_integer_scalar(block_size, name="block_size")
    validated_resamples = _positive_integer_scalar(resamples, name="resamples")
    validated_confidence = _confidence_scalar(confidence)
    return _ORIGINAL_BLOCK_BOOTSTRAP_INTERVAL(
        values,
        metric=metric,
        block_size=validated_block_size,
        resamples=validated_resamples,
        confidence=validated_confidence,
        seed=seed,
    )


def paired_delta_summary(
    delta_frame: pd.DataFrame,
    *,
    block_size: int = 50,
    resamples: int = 2000,
    seed: int | None = 0,
) -> dict[str, object]:
    """Summarize paired deltas after validating bootstrap controls."""

    return _ORIGINAL_PAIRED_DELTA_SUMMARY(
        delta_frame,
        block_size=_positive_integer_scalar(block_size, name="block_size"),
        resamples=_positive_integer_scalar(resamples, name="resamples"),
        seed=seed,
    )


def track_purity_summary(
    selected_radar: pd.DataFrame,
    *,
    track_column: str = "track_id",
) -> dict[str, float | int | None]:
    """Return track purity without truncating or rounding track identifiers."""

    if selected_radar.empty or track_column not in selected_radar.columns:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }

    track_ids = [
        track_id
        for value in selected_radar[track_column].tolist()
        if (track_id := optional_int(value)) is not None
    ]
    if not track_ids:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }

    counts = pd.Series(track_ids, dtype=object).value_counts(sort=True)
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    return {
        "selected_radar_rows": int(len(selected_radar)),
        "dominant_track_id": int(counts.index[0]),
        "dominant_track_fraction": float(probabilities[0]),
        "selected_track_entropy": _IMPL._entropy(probabilities),
        "selected_track_count": int(len(counts)),
    }


_IMPL.block_bootstrap_interval = block_bootstrap_interval
_IMPL.paired_delta_summary = paired_delta_summary
_IMPL.track_purity_summary = track_purity_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_integer_scalar"] = _positive_integer_scalar
globals()["_confidence_scalar"] = _confidence_scalar
globals()["block_bootstrap_interval"] = block_bootstrap_interval
globals()["paired_delta_summary"] = paired_delta_summary
globals()["track_purity_summary"] = track_purity_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
