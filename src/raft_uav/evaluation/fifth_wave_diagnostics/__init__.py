"""Compatibility fixes for fifth-wave reliability diagnostics.

The maintained implementation lives in the sibling ``fifth_wave_diagnostics.py``
module. This package preserves the public import path while keeping serialized
track identifiers exact, validating bootstrap controls before empty-input
returns or lossy integer coercion, and keeping nearest-time error alignment
inside each independent sequence.
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
_ORIGINAL_PAIRED_ERROR_DELTA_FRAME = _IMPL.paired_error_delta_frame
_ORIGINAL_ALIGNED_ERROR_COMPONENTS = _IMPL._aligned_error_components
_ORIGINAL_ESTIMATE_ERROR_FRAME = _IMPL.estimate_error_frame
_MISSING_SEQUENCE_ID_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


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


def _normalize_sequence_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return exact non-missing sequence identifiers for local alignment."""

    rows = pd.DataFrame(frame).copy()
    raw = rows["sequence_id"]
    missing = raw.isna()
    text = raw.where(~missing, "").astype(str).str.strip()
    invalid = missing | text.str.casefold().isin(_MISSING_SEQUENCE_ID_TEXT)
    if bool(invalid.any()):
        examples = invalid.index[invalid].tolist()[:5]
        raise ValueError(
            f"{name}.sequence_id contains missing or blank values at rows {examples}"
        )
    rows["sequence_id"] = text
    return rows


def _sequence_local_frames(
    **frames: pd.DataFrame,
) -> dict[str, pd.DataFrame] | None:
    """Normalize sequence metadata or preserve legacy single-sequence behavior."""

    materialized = {
        name: pd.DataFrame(frame).copy()
        for name, frame in frames.items()
    }
    presence = {
        name: "sequence_id" in frame.columns
        for name, frame in materialized.items()
    }
    if not any(presence.values()):
        return None
    missing = sorted(name for name, present in presence.items() if not present)
    if missing:
        raise ValueError(
            "sequence_id must be present in every aligned frame when supplied; "
            f"missing from {missing}"
        )
    return {
        name: _normalize_sequence_frame(frame, name=name)
        for name, frame in materialized.items()
    }


def paired_error_delta_frame(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
    dimensions: int = 3,
    label_a: str = "method_a",
    label_b: str = "method_b",
) -> pd.DataFrame:
    """Return paired errors without matching equal timestamps across sequences."""

    frames = _sequence_local_frames(
        method_a=method_a,
        method_b=method_b,
        truth=truth,
    )
    if frames is None:
        return _ORIGINAL_PAIRED_ERROR_DELTA_FRAME(
            method_a,
            method_b,
            truth,
            max_time_delta_s=max_time_delta_s,
            dimensions=dimensions,
            label_a=label_a,
            label_b=label_b,
        )

    method_a_rows = frames["method_a"]
    method_b_rows = frames["method_b"]
    truth_rows = frames["truth"]
    _IMPL._validate_position_frame(method_a_rows, "method_a")
    _IMPL._validate_position_frame(method_b_rows, "method_b")
    _IMPL._validate_position_frame(truth_rows, "truth")
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")

    parts: list[pd.DataFrame] = []
    for sequence_id, sequence_truth in truth_rows.groupby(
        "sequence_id",
        sort=True,
    ):
        sequence_a = method_a_rows.loc[
            method_a_rows["sequence_id"] == sequence_id
        ]
        sequence_b = method_b_rows.loc[
            method_b_rows["sequence_id"] == sequence_id
        ]
        part = _ORIGINAL_PAIRED_ERROR_DELTA_FRAME(
            sequence_a,
            sequence_b,
            sequence_truth,
            max_time_delta_s=max_time_delta_s,
            dimensions=dimensions,
            label_a=label_a,
            label_b=label_b,
        )
        part.insert(0, "sequence_id", str(sequence_id))
        parts.append(part)
    if parts:
        return pd.concat(parts, ignore_index=True, sort=False)

    empty = _ORIGINAL_PAIRED_ERROR_DELTA_FRAME(
        method_a_rows,
        method_b_rows,
        truth_rows,
        max_time_delta_s=max_time_delta_s,
        dimensions=dimensions,
        label_a=label_a,
        label_b=label_b,
    )
    empty.insert(0, "sequence_id", pd.Series(dtype=str))
    return empty


def _aligned_error_components(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Align truth rows only to estimates from the same sequence."""

    frames = _sequence_local_frames(estimates=estimates, truth=truth)
    if frames is None:
        return _ORIGINAL_ALIGNED_ERROR_COMPONENTS(
            estimates,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    estimate_rows = frames["estimates"]
    truth_rows = frames["truth"]
    parts: list[pd.DataFrame] = []
    for sequence_id, sequence_truth in truth_rows.groupby(
        "sequence_id",
        sort=True,
    ):
        sequence_estimates = estimate_rows.loc[
            estimate_rows["sequence_id"] == sequence_id
        ]
        part = _ORIGINAL_ALIGNED_ERROR_COMPONENTS(
            sequence_estimates,
            sequence_truth,
            max_time_delta_s=max_time_delta_s,
        )
        part.insert(0, "sequence_id", str(sequence_id))
        parts.append(part)
    if parts:
        return pd.concat(parts, ignore_index=True, sort=False)

    empty = _ORIGINAL_ALIGNED_ERROR_COMPONENTS(
        estimate_rows,
        truth_rows,
        max_time_delta_s=max_time_delta_s,
    )
    empty.insert(0, "sequence_id", pd.Series(dtype=str))
    return empty


def estimate_error_frame(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> pd.DataFrame:
    """Return per-estimate errors using truth from the same sequence only."""

    frames = _sequence_local_frames(estimates=estimates, truth=truth)
    if frames is None:
        return _ORIGINAL_ESTIMATE_ERROR_FRAME(
            estimates,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    estimate_rows = frames["estimates"]
    truth_rows = frames["truth"]
    parts: list[pd.DataFrame] = []
    for sequence_id, sequence_estimates in estimate_rows.groupby(
        "sequence_id",
        sort=True,
    ):
        sequence_truth = truth_rows.loc[
            truth_rows["sequence_id"] == sequence_id
        ]
        parts.append(
            _ORIGINAL_ESTIMATE_ERROR_FRAME(
                sequence_estimates,
                sequence_truth,
                max_time_delta_s=max_time_delta_s,
            )
        )
    if parts:
        return pd.concat(parts, ignore_index=True, sort=False)
    return _ORIGINAL_ESTIMATE_ERROR_FRAME(
        estimate_rows,
        truth_rows,
        max_time_delta_s=max_time_delta_s,
    )


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


_IMPL.paired_error_delta_frame = paired_error_delta_frame
_IMPL._aligned_error_components = _aligned_error_components
_IMPL.estimate_error_frame = estimate_error_frame
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
globals()["_normalize_sequence_frame"] = _normalize_sequence_frame
globals()["_sequence_local_frames"] = _sequence_local_frames
globals()["paired_error_delta_frame"] = paired_error_delta_frame
globals()["_aligned_error_components"] = _aligned_error_components
globals()["estimate_error_frame"] = estimate_error_frame
globals()["block_bootstrap_interval"] = block_bootstrap_interval
globals()["paired_delta_summary"] = paired_delta_summary
globals()["track_purity_summary"] = track_purity_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
