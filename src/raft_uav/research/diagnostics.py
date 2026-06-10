"""Higher-level diagnostics for association, coverage, leakage, and domain shift.

These helpers are deliberately independent from the command-line runners so they
can be used in notebooks, paper-table scripts, or regression tests.  They assume
normalized ENU coordinates and timestamps, matching the artifacts already written
by RaFT-UAV baselines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

PositionColumns = ("east_m", "north_m", "up_m")


@dataclass(frozen=True)
class LeakageViolation:
    """One suspicious reference to a held-out flight in a training artifact."""

    path: str
    value: str
    reason: str


def candidate_set_recall(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    distance_gate_m: float = 150.0,
    max_time_delta_s: float = 1.0,
    preselector: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Measure whether the true UAV is present in each radar candidate set.

    ``preselector`` can emulate a range gate, class-probability gate, top-K rule,
    or any other candidate-pool restriction.  The output is one row per radar
    frame with the best available candidate error and a Boolean recall flag.
    """

    _require_columns(radar, {"time_s", *PositionColumns}, "radar")
    _require_columns(truth, {"time_s", *PositionColumns}, "truth")
    rows: list[dict[str, object]] = []
    for event_key, frame in _radar_frame_groups(radar):
        pool = frame.copy()
        original_count = int(len(pool))
        if preselector is not None:
            pool = preselector(pool).copy()
        time_s = float(frame["time_s"].median())
        truth_xyz, truth_dt_s = _nearest_truth_position(
            truth,
            time_s=time_s,
            max_time_delta_s=max_time_delta_s,
        )
        row: dict[str, object] = {
            "event_key": _event_key_to_string(event_key),
            "time_s": time_s,
            "candidate_count_raw": original_count,
            "candidate_count": int(len(pool)),
            "truth_time_delta_s": truth_dt_s,
            "best_candidate_error_m": float("nan"),
            "target_present": False,
        }
        if truth_xyz is not None and not pool.empty:
            positions = pool.loc[:, PositionColumns].to_numpy(dtype=float)
            errors = np.linalg.norm(positions - truth_xyz.reshape(1, 3), axis=1)
            finite = errors[np.isfinite(errors)]
            if finite.size:
                best = float(np.min(finite))
                row["best_candidate_error_m"] = best
                row["target_present"] = bool(best <= float(distance_gate_m))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def association_regret(
    selected: pd.DataFrame,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 1.0,
) -> pd.DataFrame:
    """Compare selected radar rows with the best available candidate per frame.

    Regret is ``selected_error_m - best_candidate_error_m``.  Positive regret is
    association loss, independent of the downstream filter.
    """

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "event_key",
                "time_s",
                "selected_error_m",
                "best_candidate_error_m",
                "association_regret_m",
                "selected_track_id",
                "best_track_id",
            ]
        )
    _require_columns(selected, {"time_s", *PositionColumns}, "selected")
    radar_by_key = {key: frame for key, frame in _radar_frame_groups(radar)}
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        key = _row_event_key(row)
        candidates = radar_by_key.get(key)
        if candidates is None:
            candidates = _nearest_radar_frame(radar, float(row["time_s"]))
            key = _radar_event_key(candidates)
        truth_xyz, truth_dt_s = _nearest_truth_position(
            truth,
            time_s=float(row["time_s"]),
            max_time_delta_s=max_time_delta_s,
        )
        out: dict[str, object] = {
            "event_key": _event_key_to_string(key),
            "time_s": float(row["time_s"]),
            "truth_time_delta_s": truth_dt_s,
            "selected_error_m": float("nan"),
            "best_candidate_error_m": float("nan"),
            "association_regret_m": float("nan"),
            "selected_track_id": _optional_int(row.get("track_id")),
            "best_track_id": None,
            "candidate_count": 0 if candidates is None else int(len(candidates)),
        }
        if truth_xyz is not None and candidates is not None and not candidates.empty:
            selected_xyz = pd.to_numeric(
                row.loc[list(PositionColumns)], errors="coerce"
            ).to_numpy(dtype=float)
            selected_error = float("nan")
            if np.isfinite(selected_xyz).all():
                selected_error = float(np.linalg.norm(selected_xyz - truth_xyz))
                out["selected_error_m"] = selected_error

            candidate_xyz = (
                candidates.loc[:, PositionColumns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
            finite_candidates = np.isfinite(candidate_xyz).all(axis=1)
            if finite_candidates.any():
                valid_indices = np.flatnonzero(finite_candidates)
                errors = np.linalg.norm(
                    candidate_xyz[finite_candidates] - truth_xyz.reshape(1, 3),
                    axis=1,
                )
                best_valid_idx = int(np.argmin(errors))
                best_idx = int(valid_indices[best_valid_idx])
                best_error = float(errors[best_valid_idx])
                out["best_candidate_error_m"] = best_error
                out["best_track_id"] = _optional_int(candidates.iloc[best_idx].get("track_id"))
                if math.isfinite(selected_error):
                    out["association_regret_m"] = selected_error - best_error
        rows.append(out)
    return pd.DataFrame.from_records(rows)


def association_regret_summary(regret: pd.DataFrame) -> dict[str, object]:
    """Summarize an association-regret frame."""

    if regret.empty or "association_regret_m" not in regret.columns:
        return {
            "association_regret_count": 0,
            "association_regret_mean_m": float("nan"),
            "association_regret_p95_m": float("nan"),
            "catastrophic_regret_count": 0,
        }
    values = pd.to_numeric(regret["association_regret_m"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "association_regret_count": 0,
            "association_regret_mean_m": float("nan"),
            "association_regret_p95_m": float("nan"),
            "catastrophic_regret_count": 0,
        }
    return {
        "association_regret_count": int(values.size),
        "association_regret_mean_m": float(np.mean(values)),
        "association_regret_p50_m": float(np.percentile(values, 50)),
        "association_regret_p95_m": float(np.percentile(values, 95)),
        "association_regret_max_m": float(np.max(values)),
        "catastrophic_regret_count": int(np.count_nonzero(values > 100.0)),
    }


def track_switch_metrics(selected: pd.DataFrame, *, long_gap_s: float = 5.0) -> dict[str, object]:
    """Return identity-stability metrics for selected radar rows."""

    if selected.empty:
        return {
            "selected_radar_rows": 0,
            "track_switch_count": 0,
            "unique_track_ids": 0,
            "dominant_track_fraction": float("nan"),
            "track_id_entropy": float("nan"),
            "long_gap_count": 0,
        }
    ordered = selected.sort_values("time_s") if "time_s" in selected.columns else selected.copy()
    track_ids = (
        pd.to_numeric(ordered["track_id"], errors="coerce")
        if "track_id" in ordered.columns
        else pd.Series([np.nan] * len(ordered))
    )
    finite = track_ids.dropna().astype(int)
    switches = 0
    previous: int | None = None
    for value in finite:
        if previous is not None and int(value) != previous:
            switches += 1
        previous = int(value)
    counts = finite.value_counts()
    probabilities = counts.to_numpy(dtype=float) / max(float(counts.sum()), 1.0)
    entropy = float(-np.sum(probabilities * np.log2(np.clip(probabilities, 1e-12, 1.0))))
    times = pd.to_numeric(ordered.get("time_s", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    gaps = np.diff(times[np.isfinite(times)])
    return {
        "selected_radar_rows": int(len(selected)),
        "track_switch_count": int(switches),
        "unique_track_ids": int(len(counts)),
        "dominant_track_fraction": float(probabilities.max()) if probabilities.size else float("nan"),
        "track_id_entropy": entropy if probabilities.size else float("nan"),
        "long_gap_count": int(np.count_nonzero(gaps > float(long_gap_s))) if gaps.size else 0,
        "max_selected_gap_s": float(np.max(gaps)) if gaps.size else 0.0,
    }


def domain_shift_summary(
    training: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare held-out numeric distributions against training distributions."""

    train = _concat_training_frames(training)
    if columns is None:
        columns = [
            col
            for col in heldout.columns
            if col in train.columns and pd.api.types.is_numeric_dtype(heldout[col])
        ]
    rows: list[dict[str, object]] = []
    for col in columns:
        a = pd.to_numeric(train[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(heldout[col], errors="coerce").dropna().to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            continue
        train_std = float(np.std(a)) or 1.0
        rows.append(
            {
                "feature": col,
                "train_count": int(a.size),
                "heldout_count": int(b.size),
                "train_mean": float(np.mean(a)),
                "heldout_mean": float(np.mean(b)),
                "mean_shift_z": float((np.mean(b) - np.mean(a)) / train_std),
                "train_p50": float(np.percentile(a, 50)),
                "heldout_p50": float(np.percentile(b, 50)),
                "train_p90": float(np.percentile(a, 90)),
                "heldout_p90": float(np.percentile(b, 90)),
                "ks_distance": _ks_distance(a, b),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["ks_distance", "feature"], ascending=[False, True]
    )


def latency_curve(
    estimates_by_latency: Mapping[float | str, pd.DataFrame],
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> pd.DataFrame:
    """Evaluate RMSE/coverage as a function of allowed processing latency."""

    _require_columns(truth, {"time_s", *PositionColumns}, "truth")
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_xyz = truth.loc[:, PositionColumns].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for latency, estimates in estimates_by_latency.items():
        if estimates.empty:
            errors = np.empty(0)
            covered = 0
        else:
            _require_columns(estimates, {"time_s", *PositionColumns}, "estimates")
            estimate_times = estimates["time_s"].to_numpy(dtype=float)
            estimate_xyz = estimates.loc[:, PositionColumns].to_numpy(dtype=float)
            nearest = _nearest_time_indices(estimate_times, truth_times)
            dt_s = np.abs(estimate_times[nearest] - truth_times)
            keep = dt_s <= float(max_time_delta_s)
            covered = int(np.count_nonzero(keep))
            errors = np.linalg.norm(estimate_xyz[nearest][keep] - truth_xyz[keep], axis=1)
        rows.append(
            {
                "latency_s": float(latency),
                "truth_rows": int(len(truth)),
                "covered_truth_rows": covered,
                "truth_coverage_rate": float(covered / len(truth)) if len(truth) else float("nan"),
                "error_3d_count": int(errors.size),
                "error_3d_rmse_m": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
                "error_3d_p95_m": float(np.percentile(errors, 95)) if errors.size else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("latency_s").reset_index(drop=True)


def leakage_sentinel(
    payload: Any,
    *,
    heldout_flight: str,
    allowed_paths: Sequence[str] = (),
) -> list[LeakageViolation]:
    """Find suspicious held-out flight references inside training metadata.

    This is intentionally conservative: references under paths containing
    ``heldout`` or listed in ``allowed_paths`` are allowed, while references
    under training/model/calibration keys are reported.
    """

    needle = str(heldout_flight).lower()
    allowed = tuple(str(path) for path in allowed_paths)
    violations: list[LeakageViolation] = []

    def visit(value: Any, path: str) -> None:
        if allowed and any(path.startswith(prefix) for prefix in allowed):
            return
        lower_path = path.lower()
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple, set)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, str) and needle in value.lower():
            if "heldout" in lower_path or "test" in lower_path:
                return
            reason = "held-out flight appears in non-test metadata"
            if any(token in lower_path for token in ("train", "model", "calib", "fit")):
                reason = "held-out flight appears in training/calibration metadata"
            violations.append(LeakageViolation(path=path, value=value, reason=reason))

    visit(payload, "")
    return violations


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _concat_training_frames(
    training: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(training, pd.DataFrame):
        return training
    if isinstance(training, Mapping):
        frames = list(training.values())
    else:
        frames = list(training)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _radar_frame_groups(radar: pd.DataFrame) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
    if radar.empty:
        return []
    group_column = "frame_index" if "frame_index" in radar.columns else "time_s"
    ordered = radar.sort_values([c for c in ("time_s", "frame_index", "track_id") if c in radar.columns])
    groups = []
    for key, group in ordered.groupby(group_column, sort=True):
        if group_column == "frame_index":
            event_key = ("frame_index", int(float(key)))
        else:
            event_key = ("time_s", round(float(group["time_s"].median()), 9))
        groups.append((event_key, group.copy()))
    return groups


def _radar_event_key(frame: pd.DataFrame) -> tuple[str, int | float]:
    if "frame_index" in frame.columns and not frame.empty:
        value = pd.to_numeric(frame["frame_index"], errors="coerce").dropna()
        if not value.empty:
            return ("frame_index", int(value.iloc[0]))
    return ("time_s", round(float(frame["time_s"].median()), 9))


def _row_event_key(row: pd.Series) -> tuple[str, int | float]:
    if "frame_index" in row.index:
        value = _optional_float(row.get("frame_index"))
        if value is not None:
            return ("frame_index", int(value))
    return ("time_s", round(float(row["time_s"]), 9))


def _event_key_to_string(key: tuple[str, int | float]) -> str:
    return f"{key[0]}:{key[1]}"


def _nearest_radar_frame(radar: pd.DataFrame, time_s: float) -> pd.DataFrame | None:
    groups = _radar_frame_groups(radar)
    if not groups:
        return None
    return min(groups, key=lambda item: abs(float(item[1]["time_s"].median()) - time_s))[1]


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_time_delta_s: float,
) -> tuple[np.ndarray | None, float]:
    times = truth["time_s"].to_numpy(dtype=float)
    if not bool(np.any(np.isfinite(times))):
        return None, float("nan")
    idx = int(_nearest_time_indices(times, np.array([float(time_s)]))[0])
    dt_s = float(abs(times[idx] - float(time_s)))
    if dt_s > float(max_time_delta_s):
        return None, dt_s
    return truth.loc[:, PositionColumns].to_numpy(dtype=float)[idx], dt_s


def _nearest_time_indices(source_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    source = np.asarray(source_times, dtype=float).reshape(-1)
    query = np.asarray(query_times, dtype=float).reshape(-1)
    finite_source = np.isfinite(source)
    if not bool(np.any(finite_source)):
        return np.zeros(query.size, dtype=int)

    original_indices = np.flatnonzero(finite_source)
    finite_values = source[finite_source]
    sort_order = np.argsort(finite_values, kind="mergesort")
    sorted_source = finite_values[sort_order]
    sorted_original_indices = original_indices[sort_order]

    insertion = np.searchsorted(sorted_source, query)
    right = np.clip(insertion, 0, sorted_source.size - 1)
    left = np.clip(insertion - 1, 0, sorted_source.size - 1)
    use_right = np.abs(sorted_source[right] - query) < np.abs(sorted_source[left] - query)
    return sorted_original_indices[np.where(use_right, right, left)]


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    values = np.sort(np.unique(np.concatenate([a, b])))
    if values.size == 0:
        return float("nan")
    cdf_a = np.searchsorted(np.sort(a), values, side="right") / max(a.size, 1)
    cdf_b = np.searchsorted(np.sort(b), values, side="right") / max(b.size, 1)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)
