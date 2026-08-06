"""Keep pooled oracle-gap diagnostics within sequence boundaries."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int


_oracle_gap = import_module("raft_uav.evaluation.oracle_gap_decomposition")
_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP = _oracle_gap.decompose_radar_oracle_gap
_ORIGINAL_SELECTED_TRACK_STABILITY_METRICS = (
    _oracle_gap.selected_track_stability_metrics
)


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return normalized sequence identifiers without numeric coercion."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.fillna("")


def _sequence_subset(
    frame: pd.DataFrame | None,
    sequence_key: str,
    *,
    allow_unscoped: bool,
) -> pd.DataFrame | None:
    """Return one sequence, failing closed for unlabeled pooled auxiliaries."""

    if frame is None or frame.empty:
        return frame
    if "sequence_id" not in frame.columns:
        return frame if allow_unscoped else None
    keys = _sequence_keys(frame["sequence_id"])
    return frame.loc[keys.eq(sequence_key)].copy()


def _placeholder_truth(
    truth: pd.DataFrame,
    radar: pd.DataFrame,
    max_delta_s: float,
) -> pd.DataFrame:
    """Build a valid truth row guaranteed to fall outside the matching gate."""

    placeholder = truth.iloc[[0]].copy()
    times = pd.to_numeric(radar.get("time_s"), errors="coerce").to_numpy(dtype=float)
    finite_times = times[np.isfinite(times)]
    anchor = float(np.max(finite_times)) if finite_times.size else 0.0
    placeholder["time_s"] = anchor + float(max_delta_s) + 1.0
    return placeholder


def decompose_radar_oracle_gap(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    config: object | None = None,
) -> pd.DataFrame:
    """Evaluate pooled radar frames independently for every sequence.

    Legacy sequence-less inputs retain the maintained implementation. When both
    radar and truth carry ``sequence_id``, local frame counters and timestamps are
    isolated before candidate, selected-radar, truth, and estimate comparisons.
    """

    radar_rows = pd.DataFrame(radar).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if (
        radar_rows.empty
        or truth_rows.empty
        or "sequence_id" not in radar_rows.columns
        or "sequence_id" not in truth_rows.columns
    ):
        return _ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP(
            radar=radar_rows,
            truth=truth_rows,
            selected_radar=selected_radar,
            estimates=estimates,
            config=config,
        )

    cfg = config or _oracle_gap.OracleGapConfig()
    radar_keys = _sequence_keys(radar_rows["sequence_id"])
    truth_keys = _sequence_keys(truth_rows["sequence_id"])
    sequence_keys = list(pd.unique(radar_keys))
    allow_unscoped_auxiliary = len(sequence_keys) == 1
    chunks: list[pd.DataFrame] = []

    for sequence_key in sequence_keys:
        radar_part = radar_rows.loc[radar_keys.eq(sequence_key)].copy()
        truth_part = truth_rows.loc[truth_keys.eq(sequence_key)].copy()
        if truth_part.empty:
            truth_part = _placeholder_truth(
                truth_rows,
                radar_part,
                float(cfg.truth_time_gate_s),
            )
        chunk = _ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP(
            radar=radar_part,
            truth=truth_part,
            selected_radar=_sequence_subset(
                selected_radar,
                sequence_key,
                allow_unscoped=allow_unscoped_auxiliary,
            ),
            estimates=_sequence_subset(
                estimates,
                sequence_key,
                allow_unscoped=allow_unscoped_auxiliary,
            ),
            config=cfg,
        )
        chunk.insert(0, "sequence_id", sequence_key)
        chunks.append(chunk)

    if not chunks:
        columns = ["sequence_id", *_oracle_gap._ORACLE_GAP_COLUMNS]
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def selected_track_stability_metrics(
    selected_radar: pd.DataFrame | None,
) -> dict[str, object]:
    """Count track transitions and time gaps only within each sequence."""

    base = dict(_ORIGINAL_SELECTED_TRACK_STABILITY_METRICS(selected_radar))
    if (
        selected_radar is None
        or selected_radar.empty
        or "track_id" not in selected_radar.columns
        or "sequence_id" not in selected_radar.columns
    ):
        return base

    selected_rows = pd.DataFrame(selected_radar).copy()
    sequence_keys = _sequence_keys(selected_rows["sequence_id"])
    switches = 0
    transition_count = 0
    gap_parts: list[np.ndarray] = []

    for sequence_key in pd.unique(sequence_keys):
        part = selected_rows.loc[sequence_keys.eq(sequence_key)].copy()
        sort_columns = [
            column
            for column in ("time_s", "frame_index")
            if column in part.columns
        ]
        ordered = part.sort_values(sort_columns) if sort_columns else part
        track_ids = pd.Series(
            [optional_int(value) for value in ordered["track_id"]],
            index=ordered.index,
            dtype="Int64",
        ).dropna()
        values = track_ids.to_numpy(dtype=int)
        if values.size > 1:
            switches += int(np.count_nonzero(values[1:] != values[:-1]))
            transition_count += int(values.size - 1)
        gaps = _oracle_gap._time_gaps_s(ordered)
        if gaps.size:
            gap_parts.append(gaps)

    all_gaps = np.concatenate(gap_parts) if gap_parts else np.empty(0, dtype=float)
    base.update(
        {
            "selected_sequence_count": int(len(pd.unique(sequence_keys))),
            "track_switch_count": int(switches),
            "track_switch_rate": _oracle_gap._safe_rate(
                switches,
                transition_count,
            ),
            "selected_time_gap_p95_s": _oracle_gap._percentile_or_nan(
                all_gaps,
                95,
            ),
            "selected_time_gap_max_s": (
                float(np.max(all_gaps)) if all_gaps.size else float("nan")
            ),
        }
    )
    return base


def install() -> None:
    """Install sequence scoping on public and legacy implementation paths."""

    if getattr(_oracle_gap, "_sequence_scope_patch_applied", False):
        return
    _oracle_gap.decompose_radar_oracle_gap = decompose_radar_oracle_gap
    _oracle_gap.selected_track_stability_metrics = selected_track_stability_metrics
    implementation = getattr(_oracle_gap, "_IMPL", None)
    if implementation is not None:
        implementation.decompose_radar_oracle_gap = decompose_radar_oracle_gap
        implementation.selected_track_stability_metrics = (
            selected_track_stability_metrics
        )
    _oracle_gap._sequence_scope_patch_applied = True


install()
