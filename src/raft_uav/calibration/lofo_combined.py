"""LOFO-safe time-offset and bias-calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import bias_training_rows, fit_bias_correction_bank
from raft_uav.diagnostics.time_offset import (
    OBJECTIVE_COLUMNS,
    catprob_candidate_pool,
    nearest_candidate_to_truth,
    radar_frame_groups,
    sweep_positions_against_truth,
    sweep_radar_against_truth,
    truth_position_at_time,
)


def fit_training_time_offset(
    items: dict[str, dict[str, pd.DataFrame]],
    train_names: list[str],
    *,
    source: str,
    taus_s: np.ndarray,
    objective: str = "p95",
    max_truth_time_delta_s: float = 2.0,
    radar_catprob_threshold: float = 0.4,
) -> tuple[float, pd.DataFrame]:
    """Fit one timestamp offset from training flights only."""

    sweeps = []
    for name in train_names:
        item = items[name]
        if source == "rf" and not item["rf"].empty:
            sweep = sweep_positions_against_truth(
                measurement_times_s=item["rf"]["time_s"].to_numpy(float),
                measurement_positions_m=item["rf"][["east_m", "north_m", "up_m"]].to_numpy(float),
                truth=item["truth"],
                taus_s=taus_s,
                dimensions=2,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )
        elif source == "radar" and not item["radar"].empty:
            sweep = sweep_radar_against_truth(
                radar=item["radar"],
                truth=item["truth"],
                taus_s=taus_s,
                dimensions=3,
                selection="catprob-oracle-nearest",
                catprob_threshold=radar_catprob_threshold,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )
        else:
            continue
        sweep = sweep.copy()
        sweep["flight"] = name
        sweeps.append(sweep)
    if not sweeps:
        return 0.0, pd.DataFrame()
    aggregate = aggregate_offset_sweeps(pd.concat(sweeps, ignore_index=True), objective)
    column = OBJECTIVE_COLUMNS[objective]
    values = pd.to_numeric(aggregate[column], errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0, aggregate
    index = np.flatnonzero(finite)[np.argmin(values[finite])]
    return float(aggregate.iloc[index]["tau_s"]), aggregate


def aggregate_offset_sweeps(sweep: pd.DataFrame, objective: str) -> pd.DataFrame:
    """Pool per-flight offset sweeps with matched-row weights."""

    column = OBJECTIVE_COLUMNS[objective]
    rows: list[dict[str, Any]] = []
    for tau_s, group in sweep.groupby("tau_s", sort=True):
        weights = pd.to_numeric(group["matched_count"], errors="coerce").fillna(0.0).to_numpy(float)
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & (weights > 0.0)
        metric = float("nan")
        if valid.any():
            metric = float(np.average(values[valid], weights=weights[valid]))
        rows.append(
            {
                "tau_s": float(tau_s),
                "flight_count": int(group["flight"].nunique()),
                "candidate_count": int(group["candidate_count"].sum()),
                "selected_count": int(group["selected_count"].sum()),
                "matched_count": int(group["matched_count"].sum()),
                column: metric,
            }
        )
    return pd.DataFrame.from_records(rows)


def apply_time_offsets(
    item: dict[str, pd.DataFrame],
    *,
    rf_tau_s: float,
    radar_tau_s: float,
) -> dict[str, pd.DataFrame]:
    """Shift RF/radar timestamps while preserving original ``time_s``."""

    return {
        "truth": item["truth"],
        "rf": shift_time(item["rf"], rf_tau_s, source="rf"),
        "radar": shift_time(item["radar"], radar_tau_s, source="radar"),
    }


def shift_time(frame: pd.DataFrame, tau_s: float, *, source: str) -> pd.DataFrame:
    out = frame.copy()
    if out.empty or "time_s" not in out.columns:
        return out
    if "time_s_uncorrected" not in out.columns:
        out["time_s_uncorrected"] = out["time_s"]
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce") + float(tau_s)
    out[f"{source}_time_offset_s"] = float(tau_s)
    return out


def fit_training_bias_bank(
    shifted_items: dict[str, dict[str, pd.DataFrame]],
    train_names: list[str],
    *,
    radar_catprob_threshold: float = 0.4,
    max_truth_time_delta_s: float = 2.0,
    max_position_error_m: float = 300.0,
    ridge_alpha: float = 1.0,
    min_samples: int = 5,
):
    """Fit RF/radar bias models from offset-corrected training flights only."""

    rows_by_source: dict[str, list[pd.DataFrame]] = {"rf": [], "radar": []}
    for name in train_names:
        item = shifted_items[name]
        if not item["rf"].empty:
            rows_by_source["rf"].append(
                bias_training_rows(
                    item["rf"],
                    item["truth"],
                    source="rf",
                    max_time_delta_s=max_truth_time_delta_s,
                    max_position_error_m=max_position_error_m,
                )
            )
        selected_radar = oracle_selected_radar_rows(
            item["radar"],
            item["truth"],
            radar_catprob_threshold=radar_catprob_threshold,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        if not selected_radar.empty:
            rows_by_source["radar"].append(
                bias_training_rows(
                    selected_radar,
                    item["truth"],
                    source="radar",
                    max_time_delta_s=max_truth_time_delta_s,
                    max_position_error_m=max_position_error_m,
                )
            )
    combined = {k: pd.concat(v, ignore_index=True) for k, v in rows_by_source.items() if v}
    return fit_bias_correction_bank(combined, ridge_alpha=ridge_alpha, min_samples=min_samples)


def oracle_selected_radar_rows(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    radar_catprob_threshold: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    rows = []
    for group in radar_frame_groups(radar):
        time_s = float(group["time_s"].median())
        truth_position = truth_position_at_time(truth, time_s, max_delta_s=max_truth_time_delta_s)
        selected = nearest_candidate_to_truth(
            catprob_candidate_pool(group, radar_catprob_threshold),
            truth_position,
        )
        if selected is not None:
            rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=radar.columns)
    return pd.DataFrame(rows).reset_index(drop=True)
