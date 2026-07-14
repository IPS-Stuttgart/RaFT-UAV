"""Compatibility package for sequence-scoped oracle-gap diagnostics.

The maintained implementation lives in the sibling
``oracle_gap_decomposition.py`` module. This package preserves the public
import path while preventing pooled flights with overlapping timestamps or
local frame indices from being evaluated against one another.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_gap_decomposition.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_gap_decomposition_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load oracle-gap implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP = _IMPL.decompose_radar_oracle_gap
_ORIGINAL_SELECTED_TRACK_STABILITY_METRICS = _IMPL.selected_track_stability_metrics


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed nullable sequence identifiers for pooled diagnostics."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    return keys.where(keys.notna() & keys.ne(""))


def _sequence_subset(
    frame: pd.DataFrame | None,
    sequence_key: str,
) -> pd.DataFrame | None:
    if frame is None or frame.empty or "sequence_id" not in frame.columns:
        return frame
    keys = _sequence_keys(frame["sequence_id"]).fillna("")
    mask = keys.eq(sequence_key).to_numpy(dtype=bool)
    return frame.iloc[np.flatnonzero(mask)].copy()


def _placeholder_truth(
    truth: pd.DataFrame,
    radar: pd.DataFrame,
    max_delta_s: float,
) -> pd.DataFrame:
    placeholder = truth.iloc[[0]].copy()
    times = pd.to_numeric(radar.get("time_s"), errors="coerce").to_numpy(dtype=float)
    finite_times = times[np.isfinite(times)]
    anchor = float(np.max(finite_times)) if finite_times.size else 0.0
    placeholder["time_s"] = anchor + max(float(max_delta_s), 1.0) + 1.0
    return placeholder


def _mark_no_truth(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["truth_available"] = False
    out["nearest_candidate_error_m"] = np.nan
    out["nearest_candidate_track_id"] = ""
    out["has_plausible_candidate"] = False
    out["selected_present"] = False
    out["selected_track_id"] = ""
    out["selected_error_m"] = np.nan
    out["selected_is_plausible"] = False
    out["selected_replay_accepted"] = ""
    out["estimate_error_m"] = np.nan
    out["category"] = "no_truth"
    return out


def decompose_radar_oracle_gap(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    selected_radar: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    config: object | None = None,
) -> pd.DataFrame:
    """Decompose pooled radar frames without crossing sequence boundaries.

    Sequence-less inputs retain the historical global behavior. When radar and
    truth both provide ``sequence_id``, each radar sequence is evaluated only
    against matching truth. Selected radar rows and estimates are scoped as
    well when they provide sequence identifiers.
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

    cfg = config or _IMPL.OracleGapConfig()
    radar_keys = _sequence_keys(radar_rows["sequence_id"]).fillna("")
    truth_keys = _sequence_keys(truth_rows["sequence_id"]).fillna("")
    chunks: list[pd.DataFrame] = []
    for sequence_key in pd.unique(radar_keys):
        radar_mask = radar_keys.eq(sequence_key).to_numpy(dtype=bool)
        radar_part = radar_rows.iloc[np.flatnonzero(radar_mask)].copy()
        truth_mask = truth_keys.eq(sequence_key).to_numpy(dtype=bool)
        truth_part = truth_rows.iloc[np.flatnonzero(truth_mask)].copy()
        truth_missing = truth_part.empty
        if truth_missing:
            truth_part = _placeholder_truth(
                truth_rows,
                radar_part,
                float(cfg.truth_time_gate_s),
            )

        chunk = _ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP(
            radar=radar_part,
            truth=truth_part,
            selected_radar=_sequence_subset(selected_radar, str(sequence_key)),
            estimates=_sequence_subset(estimates, str(sequence_key)),
            config=cfg,
        )
        if truth_missing:
            chunk = _mark_no_truth(chunk)
        chunk.insert(0, "sequence_id", str(sequence_key))
        chunks.append(chunk)

    if not chunks:
        columns = ["sequence_id", *_IMPL._ORACLE_GAP_COLUMNS]
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True)


def selected_track_stability_metrics(
    selected_radar: pd.DataFrame | None,
) -> dict[str, object]:
    """Count track switches and time gaps only within each sequence."""

    base = dict(_ORIGINAL_SELECTED_TRACK_STABILITY_METRICS(selected_radar))
    if (
        selected_radar is None
        or selected_radar.empty
        or "track_id" not in selected_radar.columns
        or "sequence_id" not in selected_radar.columns
    ):
        return base

    selected_rows = pd.DataFrame(selected_radar).copy()
    sequence_keys = _sequence_keys(selected_rows["sequence_id"]).fillna("")
    switches = 0
    transition_count = 0
    gap_parts: list[np.ndarray] = []
    for sequence_key in pd.unique(sequence_keys):
        mask = sequence_keys.eq(sequence_key).to_numpy(dtype=bool)
        part = selected_rows.iloc[np.flatnonzero(mask)].copy()
        sort_columns = [c for c in ("time_s", "frame_index") if c in part.columns]
        ordered = part.sort_values(sort_columns) if sort_columns else part
        track_ids = pd.to_numeric(ordered["track_id"], errors="coerce").dropna().astype(int)
        values = track_ids.to_numpy(dtype=int)
        if values.size > 1:
            switches += int(np.count_nonzero(values[1:] != values[:-1]))
            transition_count += int(values.size - 1)
        gaps = _IMPL._time_gaps_s(ordered)
        if gaps.size:
            gap_parts.append(gaps)

    all_gaps = np.concatenate(gap_parts) if gap_parts else np.empty(0, dtype=float)
    base.update(
        {
            "selected_sequence_count": int(len(pd.unique(sequence_keys))),
            "track_switch_count": int(switches),
            "track_switch_rate": _IMPL._safe_rate(switches, transition_count),
            "selected_time_gap_p95_s": _IMPL._percentile_or_nan(all_gaps, 95),
            "selected_time_gap_max_s": (
                float(np.max(all_gaps)) if all_gaps.size else float("nan")
            ),
        }
    )
    return base


_IMPL.decompose_radar_oracle_gap = decompose_radar_oracle_gap
_IMPL.selected_track_stability_metrics = selected_track_stability_metrics

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_sequence_keys"] = _sequence_keys
globals()["decompose_radar_oracle_gap"] = decompose_radar_oracle_gap
globals()["selected_track_stability_metrics"] = selected_track_stability_metrics

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
