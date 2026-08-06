"""Keep pooled oracle-gap diagnostics within sequence boundaries."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd


_oracle_gap = import_module("raft_uav.evaluation.oracle_gap_decomposition")
_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP = _oracle_gap.decompose_radar_oracle_gap


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


def install() -> None:
    """Install sequence scoping on public and legacy implementation paths."""

    if getattr(_oracle_gap, "_sequence_scope_patch_applied", False):
        return
    _oracle_gap.decompose_radar_oracle_gap = decompose_radar_oracle_gap
    implementation = getattr(_oracle_gap, "_IMPL", None)
    if implementation is not None:
        implementation.decompose_radar_oracle_gap = decompose_radar_oracle_gap
    _oracle_gap._sequence_scope_patch_applied = True


install()
