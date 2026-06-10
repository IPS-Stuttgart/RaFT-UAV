"""Comprehensive leakage-safe result-improvement diagnostics.

This module collects the interventions that are most likely to improve RaFT-UAV
results after the current baselines are already in place: candidate-set recall,
association regret, class-probability calibration, do-no-harm radar policy
shadow decisions, adaptive process-noise recommendations, and timestamp-bias
sweeps.  All functions are truth-using diagnostics intended for offline
training-fold analysis or held-out reporting; they should not be queried by an
online tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_update_policy import (
    RadarUpdatePolicy,
    classify_radar_update_row,
)
from raft_uav.io.aerpaw import (
    DEFAULT_RADAR_CLOCK_OFFSET_S,
    DEFAULT_RF_CLOCK_OFFSET_S,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)


@dataclass(frozen=True)
class NormalizedFlightFrames:
    """Normalized truth/RF/radar frames for one flight."""

    flight: str
    truth: pd.DataFrame
    rf: pd.DataFrame
    radar: pd.DataFrame


def load_normalized_flight_frames(
    dataset_root: Path,
    flight_name: str,
    *,
    rf_clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
    radar_clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> NormalizedFlightFrames:
    """Load and normalize one Dryad/AERPAW flight."""

    flight = select_flight(Path(dataset_root), flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))
    rf = (
        normalize_rf(
            read_rf_csv(flight.rf_csv),
            projector,
            origin_time,
            clock_offset_s=rf_clock_offset_s,
        )
        if flight.rf_csv is not None
        else pd.DataFrame()
    )
    radar = (
        normalize_radar(
            read_radar_tracks_json(flight.radar_json),
            projector,
            origin_time,
            clock_offset_s=radar_clock_offset_s,
        )
        if flight.radar_json is not None
        else pd.DataFrame()
    )
    return NormalizedFlightFrames(flight=flight.name, truth=truth, rf=rf, radar=radar)


def candidate_recall_regret_table(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    selected_radar: pd.DataFrame | None = None,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    catprob_threshold: float | None = None,
    top_k: int = 8,
) -> pd.DataFrame:
    """Return per-radar-frame candidate recall and selected-association regret."""

    selected_by_key = _selected_rows_by_key(selected_radar)
    rows: list[dict[str, object]] = []
    for frame in _radar_frame_groups(radar):
        time_s = float(pd.to_numeric(frame["time_s"], errors="coerce").median())
        key = _frame_key(frame)
        truth_position, truth_dt_s = _nearest_truth_position(
            truth,
            time_s=time_s,
            max_delta_s=truth_time_gate_s,
        )
        base: dict[str, object] = {
            "frame_key_type": key[0],
            "frame_key": key[1],
            "time_s": time_s,
            "candidate_rows": int(len(frame)),
            "truth_available": truth_position is not None,
            "truth_time_delta_s": np.nan if truth_dt_s is None else float(truth_dt_s),
        }
        if truth_position is None:
            rows.append({**base, "failure_bucket": "no_nearby_truth"})
            continue

        errors = _candidate_truth_errors(frame, truth_position)
        best_index = int(np.argmin(errors)) if np.isfinite(errors).any() else -1
        best_error = float(errors[best_index]) if best_index >= 0 else float("nan")
        candidate_available = bool(best_index >= 0 and best_error <= float(truth_gate_m))
        best_rank_by_catprob = _rank_of_index_by_column(frame, best_index, "cat_prob_uav", descending=True)
        selected = selected_by_key.get(key)
        selected_error = _selected_error_m(selected, truth_position)
        selected_track_id = _finite_int(selected.get("track_id")) if selected is not None else None
        if selected_error is None:
            regret = float("nan")
        else:
            regret = float(selected_error - best_error)

        catprob_recall = np.nan
        catprob_lost = False
        if catprob_threshold is not None and "cat_prob_uav" in frame.columns:
            catprob = pd.to_numeric(frame["cat_prob_uav"], errors="coerce").fillna(0.0).to_numpy()
            keep = catprob >= float(catprob_threshold)
            catprob_recall = bool(np.any((errors <= float(truth_gate_m)) & keep))
            catprob_lost = bool(candidate_available and not catprob_recall)

        rows.append(
            {
                **base,
                "best_candidate_error_m": best_error,
                "candidate_available": candidate_available,
                "candidate_recall_after_catprob": catprob_recall,
                "correct_candidate_lost_by_catprob": catprob_lost,
                "best_candidate_rank_by_catprob": best_rank_by_catprob,
                "best_candidate_in_top_k_by_catprob": (
                    np.nan
                    if np.isnan(best_rank_by_catprob)
                    else bool(best_rank_by_catprob < int(top_k))
                ),
                "selected_available": selected is not None,
                "selected_error_m": np.nan if selected_error is None else selected_error,
                "association_regret_m": regret,
                "selected_track_id": np.nan if selected_track_id is None else selected_track_id,
                "failure_bucket": _failure_bucket(candidate_available, selected_error, best_error, truth_gate_m),
            }
        )
    return pd.DataFrame(rows)


def _candidate_truth_errors(frame: pd.DataFrame, truth_position: np.ndarray) -> np.ndarray:
    positions = frame[["east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = np.isfinite(positions).all(axis=1)
    errors = np.full(len(frame), np.inf, dtype=float)
    if finite.any():
        errors[finite] = np.linalg.norm(
            positions[finite] - truth_position.reshape(1, 3),
            axis=1,
        )
    return errors


def summarize_candidate_recall_regret(table: pd.DataFrame) -> dict[str, object]:
    """Summarize candidate recall and association-regret buckets."""

    if table.empty:
        return {
            "radar_frames": 0,
            "candidate_recall_rate": float("nan"),
            "selected_recall_rate": float("nan"),
        }
    eligible = table.loc[table["truth_available"].fillna(False)]
    if eligible.empty:
        return {"radar_frames": int(len(table)), "truth_matched_frames": 0}
    out: dict[str, object] = {
        "radar_frames": int(len(table)),
        "truth_matched_frames": int(len(eligible)),
        "candidate_recall_rate": _mean_bool(eligible.get("candidate_available")),
        "selected_recall_rate": _mean_bool(
            pd.to_numeric(eligible.get("selected_error_m"), errors="coerce")
            <= pd.to_numeric(eligible.get("best_candidate_error_m"), errors="coerce").fillna(np.inf)
            + 1.0e-9
        ),
        "wrong_or_missing_association_rate": _mean_bool(
            eligible.get("failure_bucket").isin(["missed_association", "wrong_association"])
        ),
        "catprob_loss_rate": _mean_bool(eligible.get("correct_candidate_lost_by_catprob")),
        "association_regret_p95_m": _finite_percentile(eligible.get("association_regret_m"), 95),
    }
    buckets = eligible.get("failure_bucket", pd.Series(dtype=object)).value_counts(dropna=False)
    for bucket, count in buckets.items():
        out[f"bucket_{bucket}"] = int(count)
    return out


def class_probability_calibration_table(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    bins: int = 10,
) -> pd.DataFrame:
    """Return binned reliability statistics for Fortem UAV class probability."""

    if radar.empty or "cat_prob_uav" not in radar.columns:
        return pd.DataFrame()
    labels: list[bool] = []
    probs: list[float] = []
    for _, row in radar.iterrows():
        time_s = _finite_float(row.get("time_s"))
        if time_s is None:
            continue
        truth_position, _ = _nearest_truth_position(truth, time_s=time_s, max_delta_s=truth_time_gate_s)
        if truth_position is None:
            continue
        position = row[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
        if not np.isfinite(position).all():
            continue
        prob = _finite_float(row.get("cat_prob_uav"))
        if prob is None:
            continue
        labels.append(bool(np.linalg.norm(position - truth_position) <= float(truth_gate_m)))
        probs.append(float(np.clip(prob, 0.0, 1.0)))
    if not probs:
        return pd.DataFrame()
    values = pd.DataFrame({"probability": probs, "label": labels})
    values["bin_id"] = np.minimum((values["probability"] * int(bins)).astype(int), int(bins) - 1)
    grouped = values.groupby("bin_id", sort=True)
    table = grouped.agg(
        count=("label", "size"),
        mean_probability=("probability", "mean"),
        empirical_positive_rate=("label", "mean"),
    ).reset_index()
    table["abs_calibration_gap"] = np.abs(
        table["empirical_positive_rate"] - table["mean_probability"]
    )
    table["ece_contribution"] = table["count"] * table["abs_calibration_gap"] / max(len(values), 1)
    return table


def summarize_probability_calibration(table: pd.DataFrame) -> dict[str, object]:
    """Summarize binned reliability statistics."""

    if table.empty:
        return {"catprob_calibration_rows": 0, "catprob_ece": float("nan")}
    return {
        "catprob_calibration_rows": int(pd.to_numeric(table["count"], errors="coerce").sum()),
        "catprob_ece": float(pd.to_numeric(table["ece_contribution"], errors="coerce").sum()),
        "catprob_max_gap": float(pd.to_numeric(table["abs_calibration_gap"], errors="coerce").max()),
    }


def do_no_harm_decision_table(
    rows: pd.DataFrame,
    *,
    policy: RadarUpdatePolicy | None = None,
) -> pd.DataFrame:
    """Return shadow do-no-harm decisions for selected radar/update rows."""

    if rows is None or rows.empty:
        return pd.DataFrame()
    policy = policy or RadarUpdatePolicy()
    out: list[dict[str, object]] = []
    source = rows["source"] if "source" in rows.columns else pd.Series(["radar"] * len(rows), index=rows.index)
    radar_mask = source.astype(str).str.lower().eq("radar").to_numpy()
    radar_rows = rows.loc[radar_mask].copy()
    for _, row in radar_rows.iterrows():
        plan = classify_radar_update_row(row, policy)
        out.append(
            {
                "time_s": _finite_float(row.get("time_s")),
                "action": plan.action,
                "reason": plan.reason,
                "covariance_scale": plan.covariance_scale,
                "nis": plan.nis,
                "anchor_nis": plan.anchor_nis,
                "entropy": plan.entropy,
                "effective_candidates": plan.effective_candidates,
                "preceding_miss_streak": plan.preceding_miss_streak,
            }
        )
    return pd.DataFrame(out)


def adaptive_process_noise_schedule(
    estimates: pd.DataFrame,
    *,
    base_acceleration_std_mps2: float = 4.0,
    min_acceleration_std_mps2: float = 1.0,
    max_acceleration_std_mps2: float = 18.0,
    rolling_window: int = 25,
    gap_s: float = 3.0,
) -> pd.DataFrame:
    """Return a diagnostic acceleration-noise schedule from rolling NIS consistency."""

    if estimates is None or estimates.empty or "nis" not in estimates.columns:
        return pd.DataFrame()
    ordered = estimates.sort_values("time_s").reset_index(drop=True).copy()
    nis = pd.to_numeric(ordered["nis"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    raw_dim = (
        ordered["measurement_dim"]
        if "measurement_dim" in ordered.columns
        else pd.Series([3.0] * len(ordered), index=ordered.index)
    )
    dim = pd.to_numeric(raw_dim, errors="coerce").fillna(3.0)
    ratio = nis / np.maximum(dim, 1.0)
    rolling = ratio.rolling(window=max(int(rolling_window), 1), min_periods=3).median()
    rolling = rolling.fillna(ratio.expanding(min_periods=1).median()).fillna(1.0)
    dt_s = pd.to_numeric(ordered["time_s"], errors="coerce").diff().fillna(0.0)
    gap_factor = np.where(dt_s > float(gap_s), 1.5, 1.0)
    factor = np.sqrt(np.clip(rolling.to_numpy(dtype=float), 0.25, 16.0)) * gap_factor
    recommended = np.clip(
        float(base_acceleration_std_mps2) * factor,
        float(min_acceleration_std_mps2),
        float(max_acceleration_std_mps2),
    )
    return pd.DataFrame(
        {
            "time_s": ordered["time_s"],
            "source": ordered.get("source", ""),
            "nis": nis,
            "measurement_dim": dim,
            "rolling_nis_per_dim": rolling,
            "delta_t_s": dt_s,
            "gap_factor": gap_factor,
            "recommended_acceleration_std_mps2": recommended,
        }
    )


def time_bias_grid_search(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    offsets_s: Sequence[float],
    max_time_delta_s: float = 2.0,
    dimensions: int = 3,
) -> pd.DataFrame:
    """Evaluate source timestamp offsets against truth residuals."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    dimensions = 2 if dimensions == 2 or "up_m" not in frame.columns else 3
    coord_cols = ["east_m", "north_m"] if dimensions == 2 else ["east_m", "north_m", "up_m"]
    if truth is None or truth.empty or "time_s" not in truth.columns:
        return _zero_count_offset_rows(source, offsets_s)
    if not all(column in truth.columns for column in coord_cols):
        return _zero_count_offset_rows(source, offsets_s)
    sensor_times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    sensor_positions = frame[coord_cols].to_numpy(dtype=float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    if truth_times.size == 0 or not bool(np.isfinite(truth_times).any()):
        return _zero_count_offset_rows(source, offsets_s)
    truth_positions = truth[coord_cols].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for offset in offsets_s:
        shifted = sensor_times + float(offset)
        indices = _nearest_time_indices(truth_times, shifted)
        dt_s = np.abs(truth_times[indices] - shifted)
        keep = np.isfinite(dt_s) & (dt_s <= float(max_time_delta_s))
        if not bool(np.any(keep)):
            rows.append({"source": source, "offset_s": float(offset), "count": 0})
            continue
        errors = np.linalg.norm(sensor_positions[keep] - truth_positions[indices[keep]], axis=1)
        errors = errors[np.isfinite(errors)]
        rows.append(
            {
                "source": source,
                "offset_s": float(offset),
                "count": int(errors.size),
                "rmse_m": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
                "mae_m": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
                "p95_m": float(np.percentile(errors, 95)) if errors.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def comprehensive_run_report(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    rf: pd.DataFrame | None = None,
    selected_radar: pd.DataFrame | None = None,
    estimates: pd.DataFrame | None = None,
    catprob_threshold: float | None = 0.4,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    offset_grid_s: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive leakage-safe improvement report."""

    candidate_table = candidate_recall_regret_table(
        radar,
        truth,
        selected_radar=selected_radar,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
        catprob_threshold=catprob_threshold,
    )
    calibration_table = class_probability_calibration_table(
        radar,
        truth,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )
    update_rows = estimates if estimates is not None and not estimates.empty else selected_radar
    policy_table = do_no_harm_decision_table(update_rows if update_rows is not None else pd.DataFrame())
    process_table = adaptive_process_noise_schedule(
        estimates if estimates is not None else pd.DataFrame()
    )

    if offset_grid_s is None:
        offset_grid_s = np.linspace(-2.0, 2.0, 81)
    radar_offset = time_bias_grid_search(
        radar,
        truth,
        source="radar",
        offsets_s=offset_grid_s,
        max_time_delta_s=max(truth_time_gate_s, 2.0),
        dimensions=3,
    )
    rf_offset = time_bias_grid_search(
        rf if rf is not None else pd.DataFrame(),
        truth,
        source="rf",
        offsets_s=offset_grid_s,
        max_time_delta_s=max(truth_time_gate_s, 2.0),
        dimensions=2,
    )

    summary: dict[str, object] = {}
    summary.update(summarize_candidate_recall_regret(candidate_table))
    summary.update(summarize_probability_calibration(calibration_table))
    summary.update(_policy_summary(policy_table))
    summary.update(_process_noise_summary(process_table))
    summary.update(_offset_summary("radar", radar_offset))
    summary.update(_offset_summary("rf", rf_offset))
    summary["recommended_next_actions"] = recommendations_from_summary(summary)
    return {
        "summary": summary,
        "tables": {
            "candidate_recall_regret": candidate_table,
            "catprob_calibration": calibration_table,
            "do_no_harm_decisions": policy_table,
            "adaptive_process_noise": process_table,
            "radar_time_bias": radar_offset,
            "rf_time_bias": rf_offset,
        },
    }


def recommendations_from_summary(summary: Mapping[str, object]) -> list[str]:
    """Return prioritized result-improvement actions from summary metrics."""

    actions: list[str] = []
    candidate_recall = _finite_float(summary.get("candidate_recall_rate"))
    catprob_loss = _finite_float(summary.get("catprob_loss_rate"))
    wrong_or_missing = _finite_float(summary.get("wrong_or_missing_association_rate"))
    ece = _finite_float(summary.get("catprob_ece"))
    dnh_skip = _finite_float(summary.get("do_no_harm_skip_or_defer_rate"))
    max_accel = _finite_float(summary.get("adaptive_process_noise_p95_mps2"))
    radar_offset = abs(_finite_float(summary.get("radar_best_offset_s")) or 0.0)

    if candidate_recall is not None and candidate_recall < 0.95:
        actions.append("Improve candidate availability first: relax catProb/range gates, increase top-K, and report candidate-set recall before tuning association.")
    if catprob_loss is not None and catprob_loss > 0.02:
        actions.append("Enable soft catProb retention and tune its below-threshold penalty under nested LOFO; hard class-probability pruning is losing plausible target rows.")
    if wrong_or_missing is not None and wrong_or_missing > 0.05:
        actions.append("Prioritize sequence-level learned association: tune Viterbi missed/switch/reacquisition costs and learned-vs-hand unary weights on training flights only.")
    if ece is not None and ece > 0.08:
        actions.append("Calibrate or re-train radar class probability / learned likelihood scores; reliability gaps are large enough to distort association costs.")
    if dnh_skip is not None and dnh_skip > 0.03:
        actions.append("Run do-no-harm radar update ablations: soften/defer high-NIS or high-ambiguity radar updates and optimize p95/p99 rather than only RMSE.")
    if max_accel is not None and max_accel > 8.0:
        actions.append("Test adaptive process-noise schedules around gaps and high-NIS intervals; the fixed acceleration noise is probably too small for hard segments.")
    if radar_offset > 0.25:
        actions.append("Revisit radar/RF time-offset calibration and consider a bounded online time-bias state initialized from LOFO calibration.")
    if not actions:
        actions.append("No single failure mode dominates; rank methods with coverage, track-switch, p95/p99, and confidence-calibration constraints.")
    return actions


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar is None or radar.empty:
        return []
    sort_columns = [column for column in ("time_s", "frame_index", "track_id", "track_index") if column in radar.columns]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def _frame_key(frame: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in frame.columns:
        values = pd.to_numeric(frame["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return "frame_index", int(values.iloc[0])
    return "time_s", round(float(pd.to_numeric(frame["time_s"], errors="coerce").median()), 9)


def _row_key(row: pd.Series) -> tuple[str, int | float]:
    if "frame_index" in row.index:
        value = _finite_float(row.get("frame_index"))
        if value is not None:
            return "frame_index", int(value)
    value = _finite_float(row.get("time_s"))
    return "time_s", float("nan") if value is None else round(value, 9)


def _selected_rows_by_key(selected_radar: pd.DataFrame | None) -> dict[tuple[str, int | float], pd.Series]:
    if selected_radar is None or selected_radar.empty:
        return {}
    out: dict[tuple[str, int | float], pd.Series] = {}
    for _, row in selected_radar.iterrows():
        out.setdefault(_row_key(row), row)
    return out


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> tuple[np.ndarray | None, float | None]:
    if truth is None or truth.empty:
        return None, None
    times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    if times.size == 0:
        return None, None
    finite_times = np.isfinite(times)
    if not bool(finite_times.any()):
        return None, None
    idx = int(_nearest_time_indices(times, np.array([float(time_s)]))[0])
    dt_s = float(abs(times[idx] - float(time_s)))
    if dt_s > float(max_delta_s):
        return None, dt_s
    return truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[idx], dt_s


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    finite_reference = np.isfinite(reference)
    if not bool(np.any(finite_reference)):
        return np.zeros(query.size, dtype=int)
    original_indices = np.flatnonzero(finite_reference)
    finite_values = reference[finite_reference]
    sort_order = np.argsort(finite_values, kind="mergesort")
    sorted_reference = finite_values[sort_order]
    sorted_original_indices = original_indices[sort_order]
    insertion = np.searchsorted(sorted_reference, query)
    right = np.clip(insertion, 0, sorted_reference.size - 1)
    left = np.clip(insertion - 1, 0, sorted_reference.size - 1)
    use_right = np.abs(sorted_reference[right] - query) < np.abs(sorted_reference[left] - query)
    return sorted_original_indices[np.where(use_right, right, left)]


def _zero_count_offset_rows(source: str, offsets_s: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"source": source, "offset_s": float(offset), "count": 0} for offset in offsets_s
    )


def _selected_error_m(selected: pd.Series | None, truth_position: np.ndarray) -> float | None:
    if selected is None:
        return None
    try:
        position = selected[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if position.shape != (3,) or not np.isfinite(position).all():
        return None
    return float(np.linalg.norm(position - truth_position))


def _failure_bucket(
    candidate_available: bool,
    selected_error: float | None,
    best_error: float,
    truth_gate_m: float,
) -> str:
    if not candidate_available:
        return "no_usable_radar_candidate"
    if selected_error is None:
        return "missed_association"
    if selected_error > float(truth_gate_m):
        return "wrong_association"
    if selected_error - best_error > 25.0:
        return "suboptimal_association"
    return "nominal"


def _rank_of_index_by_column(
    frame: pd.DataFrame,
    index: int,
    column: str,
    *,
    descending: bool,
) -> float:
    if index < 0 or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").fillna(-np.inf if descending else np.inf)
    order = np.argsort((-values if descending else values).to_numpy(dtype=float))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(len(order), dtype=float)
    return float(ranks[index])


def _policy_summary(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        return {"do_no_harm_decision_rows": 0, "do_no_harm_skip_or_defer_rate": float("nan")}
    actions = table["action"].astype(str)
    out: dict[str, object] = {
        "do_no_harm_decision_rows": int(len(table)),
        "do_no_harm_skip_or_defer_rate": float(actions.isin(["skip", "defer"]).mean()),
        "do_no_harm_soften_rate": float(actions.eq("soften").mean()),
    }
    for action, count in actions.value_counts().items():
        out[f"do_no_harm_action_{action}"] = int(count)
    return out


def _process_noise_summary(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        return {"adaptive_process_noise_rows": 0}
    values = pd.to_numeric(table["recommended_acceleration_std_mps2"], errors="coerce").dropna()
    if values.empty:
        return {"adaptive_process_noise_rows": int(len(table))}
    return {
        "adaptive_process_noise_rows": int(len(table)),
        "adaptive_process_noise_mean_mps2": float(values.mean()),
        "adaptive_process_noise_p95_mps2": float(values.quantile(0.95)),
        "adaptive_process_noise_max_mps2": float(values.max()),
    }


def _offset_summary(prefix: str, table: pd.DataFrame) -> dict[str, object]:
    if table.empty or "rmse_m" not in table.columns:
        return {f"{prefix}_time_bias_rows": 0}
    valid = table.loc[pd.to_numeric(table["count"], errors="coerce").fillna(0) > 0].copy()
    if valid.empty:
        return {f"{prefix}_time_bias_rows": int(len(table))}
    best = valid.loc[pd.to_numeric(valid["rmse_m"], errors="coerce").idxmin()]
    return {
        f"{prefix}_time_bias_rows": int(len(table)),
        f"{prefix}_best_offset_s": float(best["offset_s"]),
        f"{prefix}_best_offset_rmse_m": float(best["rmse_m"]),
        f"{prefix}_best_offset_count": int(best["count"]),
    }


def _mean_bool(values: Any) -> float:
    if values is None:
        return float("nan")
    series = pd.Series(values).dropna()
    return float(series.astype(bool).mean()) if not series.empty else float("nan")


def _finite_percentile(values: Any, percentile: float) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    series = series[np.isfinite(series)]
    return float(np.percentile(series, percentile)) if not series.empty else float("nan")


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)
