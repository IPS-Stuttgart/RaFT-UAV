"""Order research track-switch diagnostics by sequence and numeric time."""

from __future__ import annotations

from functools import wraps
import math
from types import ModuleType
from typing import Callable

import pandas as pd

from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_orders_track_switch_times_numerically"
_SEQUENCE_KEY_COLUMN = "_raft_uav_track_switch_sequence_key"


def _sequence_key(value: object) -> str | None:
    """Normalize a scalar sequence identifier for diagnostic grouping."""

    if value is None or not pd.api.types.is_scalar(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _track_counts(module: ModuleType, selected: pd.DataFrame) -> list[int]:
    """Return per-track row counts after the public exact-ID normalization."""

    if "track_id" not in selected.columns:
        return []
    normalizer = getattr(module, "_dense_track_id_surrogates", None)
    if normalizer is None:  # pragma: no cover - compatibility fallback
        return []
    normalized = normalizer(selected["track_id"])
    return [int(value) for value in normalized.dropna().value_counts().tolist()]


def _aggregate_sequence_metrics(
    module: ModuleType,
    original: Callable[..., dict[str, object]],
    selected: pd.DataFrame,
    *,
    long_gap_s: float,
) -> dict[str, object]:
    """Aggregate track-switch diagnostics without crossing sequence boundaries."""

    sequence_keys = [_sequence_key(value) for value in selected["sequence_id"]]
    explicit_sequences = {key for key in sequence_keys if key is not None}
    if len(explicit_sequences) <= 1:
        return original(selected, long_gap_s=long_gap_s)

    normalized = selected.copy()
    normalized[_SEQUENCE_KEY_COLUMN] = [
        ("sequence", key) if key is not None else ("missing", "")
        for key in sequence_keys
    ]

    track_counts: list[int] = []
    switch_count = 0
    long_gap_count = 0
    max_selected_gap_s = 0.0
    for _, group in normalized.groupby(_SEQUENCE_KEY_COLUMN, sort=False):
        scoped = group.drop(columns=_SEQUENCE_KEY_COLUMN)
        metrics = original(scoped, long_gap_s=long_gap_s)
        switch_count += int(metrics["track_switch_count"])
        long_gap_count += int(metrics["long_gap_count"])
        max_selected_gap_s = max(
            max_selected_gap_s,
            float(metrics.get("max_selected_gap_s", 0.0)),
        )
        track_counts.extend(_track_counts(module, scoped))

    total_track_rows = sum(track_counts)
    if total_track_rows:
        probabilities = [count / total_track_rows for count in track_counts]
        dominant_track_fraction = max(probabilities)
        track_id_entropy = -sum(
            probability * math.log2(probability)
            for probability in probabilities
        )
    else:
        dominant_track_fraction = float("nan")
        track_id_entropy = float("nan")

    return {
        "selected_radar_rows": int(len(selected)),
        "track_switch_count": switch_count,
        "unique_track_ids": len(track_counts),
        "dominant_track_fraction": dominant_track_fraction,
        "track_id_entropy": track_id_entropy,
        "long_gap_count": long_gap_count,
        "max_selected_gap_s": max_selected_gap_s,
    }


def apply_diagnostics_numeric_time_patch(module: ModuleType) -> None:
    """Patch track-switch metrics to normalize timestamps and sequence scope."""

    original: Callable[..., dict[str, object]] = module.track_switch_metrics
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def track_switch_metrics(
        selected: pd.DataFrame,
        *,
        long_gap_s: float = 5.0,
    ) -> dict[str, object]:
        normalized = pd.DataFrame(selected).copy()
        if "time_s" in normalized.columns:
            normalized["time_s"] = pd.Series(
                [optional_float(value) for value in normalized["time_s"]],
                index=normalized.index,
                dtype=float,
            )
        if "sequence_id" in normalized.columns:
            return _aggregate_sequence_metrics(
                module,
                original,
                normalized,
                long_gap_s=long_gap_s,
            )
        return original(normalized, long_gap_s=long_gap_s)

    setattr(track_switch_metrics, _PATCH_MARKER, True)
    module.track_switch_metrics = track_switch_metrics
