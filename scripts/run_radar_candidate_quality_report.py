#!/usr/bin/env python3
"""Build radar-candidate quality diagnostics for AERPAW UAV tracking.

The report compares what the radar offered against what a tracker selected:

* nearest-candidate oracle error over time;
* optional selected-radar errors from baseline/tracklet outputs;
* RF measurement error over time;
* per-frame radar candidate counts and selected track-ID switches.

No model is trained here.  The script is meant to diagnose whether bad tails are
caused by unavailable radar candidates, wrong association, RF degradation, or
track-ID fragmentation.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import (
    interpolate_truth_positions,
    nearest_candidate_oracle,
    summarize_oracle_selection,
)
from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar_candidate_quality"))
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--oracle-time-offset-s", type=float, default=0.0)
    parser.add_argument("--catprob-threshold", type=float, default=0.4)
    parser.add_argument(
        "--selected-radar-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Directory containing <flight>/selected_radar.csv, or a direct selected_radar.csv. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--selected-radar-csv",
        action="append",
        default=[],
        help="Optional LABEL=PATH or PATH to a selected_radar.csv artifact. Can be repeated.",
    )
    parser.add_argument(
        "--include-catprob-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include a simple top-catProb-per-frame radar selection diagnostic",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write PNG plots in addition to CSV/JSON artifacts",
    )
    args = parser.parse_args()

    flight_names = requested_flights(args.dataset_root, args.flight)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for flight_name in flight_names:
        summary_rows.extend(run_one_flight(args, flight_name))

    summary = pd.DataFrame.from_records(summary_rows)
    summary_path = args.output_dir / "radar_candidate_quality_summary.csv"
    summary.to_csv(summary_path, index=False)
    json_path = args.output_dir / "radar_candidate_quality_summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"summary_csv={summary_path}")
    return 0


def requested_flights(dataset_root: Path, requested: Sequence[str] | None) -> list[str]:
    if requested:
        return [select_flight(dataset_root, name).name for name in requested]
    discovered = [flight.name for flight in discover_flights(dataset_root) if flight.truth_txt]
    preferred = [name for name in ("Opt1", "Opt2", "Opt3") if _matches_discovered(name, discovered)]
    return preferred or discovered


def run_one_flight(args: argparse.Namespace, flight_name: str) -> list[dict[str, object]]:
    flight = select_flight(args.dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth, projector, origin_time = normalize_truth(read_truth(flight.truth_txt))

    radar = pd.DataFrame()
    if flight.radar_json is not None:
        radar = normalize_radar(read_radar_tracks_json(flight.radar_json), projector, origin_time)
        radar = inside_truth_window(radar, truth)
    rf = pd.DataFrame()
    if flight.rf_csv is not None:
        rf = normalize_rf(read_rf_csv(flight.rf_csv), projector, origin_time)
        rf = inside_truth_window(rf, truth)

    output_dir = args.output_dir / flight.name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_quality = radar_frame_quality(radar)
    frame_quality.to_csv(output_dir / "radar_frame_quality.csv", index=False)

    oracle = nearest_candidate_oracle(
        radar,
        truth,
        time_offset_s=args.oracle_time_offset_s,
        max_time_delta_s=args.max_time_delta_s,
    )
    oracle = rename_oracle_errors(oracle)
    oracle.to_csv(output_dir / "nearest_candidate_oracle.csv", index=False)

    rf_errors = measurement_errors(rf, truth, method="rf", max_time_delta_s=args.max_time_delta_s)
    rf_errors.to_csv(output_dir / "rf_errors.csv", index=False)

    selected_errors: list[pd.DataFrame] = []
    if args.include_catprob_selection and not radar.empty:
        selected_errors.append(
            selected_error_frame(
                top_catprob_selection(radar, threshold=args.catprob_threshold),
                truth,
                method=f"catprob-top-{args.catprob_threshold:g}",
                max_time_delta_s=args.max_time_delta_s,
            )
        )
    for label, path in selected_radar_paths(args, flight.name):
        selected = pd.read_csv(path)
        selected_errors.append(
            selected_error_frame(
                selected,
                truth,
                method=label,
                max_time_delta_s=args.max_time_delta_s,
            )
        )

    if selected_errors:
        pd.concat(selected_errors, ignore_index=True).to_csv(
            output_dir / "selected_radar_errors.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "selected_radar_errors.csv", index=False)

    summary_rows = flight_summary_rows(
        flight_name=flight.name,
        radar=radar,
        oracle=oracle,
        rf_errors=rf_errors,
        selected_errors=selected_errors,
    )
    (output_dir / "radar_candidate_quality_summary.json").write_text(
        json.dumps(summary_rows, indent=2), encoding="utf-8"
    )

    if args.plots:
        write_plots(
            output_dir=output_dir,
            flight_name=flight.name,
            frame_quality=frame_quality,
            oracle=oracle,
            rf_errors=rf_errors,
            selected_errors=selected_errors,
        )
    return summary_rows


def radar_frame_quality(radar: pd.DataFrame) -> pd.DataFrame:
    if radar.empty:
        return pd.DataFrame(
            columns=[
                "frame_index",
                "time_s",
                "candidate_rows",
                "unique_track_ids",
                "cat_prob_uav_mean",
                "cat_prob_uav_max",
            ]
        )
    rows: list[dict[str, object]] = []
    for key, group in radar_frame_groups(radar):
        catprob = numeric_column(group, "cat_prob_uav")
        track_ids = numeric_column(group, "track_id")
        rows.append(
            {
                "frame_index": key,
                "time_s": float(pd.to_numeric(group["time_s"], errors="coerce").median()),
                "candidate_rows": int(len(group)),
                "unique_track_ids": int(np.unique(track_ids[np.isfinite(track_ids)]).size),
                "cat_prob_uav_mean": finite_mean(catprob),
                "cat_prob_uav_max": finite_max(catprob),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def top_catprob_selection(radar: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in radar_frame_groups(radar):
        pool = group.copy()
        if "cat_prob_uav" in pool.columns:
            catprob = pd.to_numeric(pool["cat_prob_uav"], errors="coerce")
            above = pool.loc[catprob >= float(threshold)]
            if not above.empty:
                pool = above
            catprob = pd.to_numeric(pool["cat_prob_uav"], errors="coerce").fillna(-np.inf)
            best = int(catprob.to_numpy(dtype=float).argmax())
        else:
            best = 0
        row = pool.iloc[best].copy()
        row["association_mode"] = "catprob-top"
        rows.append(row)
    if not rows:
        return radar.iloc[0:0].copy()
    return pd.DataFrame(rows).sort_values("time_s").reset_index(drop=True)


def selected_error_frame(
    selected: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    method: str,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    errors = measurement_errors(selected, truth, method=method, max_time_delta_s=max_time_delta_s)
    errors["track_switch"] = track_switch_flags(errors)
    return errors


def measurement_errors(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    method: str,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    columns = [
        "method",
        "time_s",
        "error_3d_m",
        "error_2d_m",
        "track_id",
        "cat_prob_uav",
        "association_mode",
        "association_score",
        "association_anchor_nis",
    ]
    if measurements.empty:
        return pd.DataFrame(columns=columns)
    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(measurements.columns):
        missing = sorted(required - set(measurements.columns))
        raise KeyError(f"measurement frame is missing required columns {missing}")
    out = measurements.copy()
    positions, valid = interpolate_truth_positions(
        truth,
        out["time_s"].to_numpy(dtype=float),
        max_time_delta_s=max_time_delta_s,
    )
    xyz = out[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    residuals = xyz - positions
    out["method"] = method
    out["error_3d_m"] = np.where(valid, np.linalg.norm(residuals, axis=1), np.nan)
    out["error_2d_m"] = np.where(valid, np.linalg.norm(residuals[:, :2], axis=1), np.nan)
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
    extra = [c for c in ("frame_index", "track_index", "oracle_candidate_rows") if c in out.columns]
    return out[[*columns, *extra]].sort_values("time_s").reset_index(drop=True)


def rename_oracle_errors(oracle: pd.DataFrame) -> pd.DataFrame:
    out = oracle.copy()
    if "oracle_error_3d_m" in out.columns:
        out["error_3d_m"] = out["oracle_error_3d_m"]
    if "oracle_error_2d_m" in out.columns:
        out["error_2d_m"] = out["oracle_error_2d_m"]
    out["method"] = "nearest-candidate-oracle"
    out["track_switch"] = track_switch_flags(out)
    return out


def flight_summary_rows(
    *,
    flight_name: str,
    radar: pd.DataFrame,
    oracle: pd.DataFrame,
    rf_errors: pd.DataFrame,
    selected_errors: Sequence[pd.DataFrame],
) -> list[dict[str, object]]:
    frame_count = radar_frame_count(radar)
    rows: list[dict[str, object]] = []
    oracle_summary = summarize_oracle_selection(oracle, frame_count=frame_count)
    rows.append(
        {
            "flight": flight_name,
            "method": "nearest-candidate-oracle",
            "source": "radar",
            "radar_frame_count": frame_count,
            "track_switch_count": int(track_switch_flags(oracle).sum()),
            **prefixless_summary(oracle_summary),
        }
    )
    rows.append(summary_from_errors(flight_name, "rf", "rf", rf_errors, frame_count=None))
    for frame in selected_errors:
        method = str(frame["method"].iloc[0]) if not frame.empty else "selected-radar"
        rows.append(
            summary_from_errors(
                flight_name, method, "selected_radar", frame, frame_count=frame_count
            )
        )
    return rows


def summary_from_errors(
    flight_name: str,
    method: str,
    source: str,
    errors: pd.DataFrame,
    *,
    frame_count: int | None,
) -> dict[str, object]:
    e3 = finite_values(errors.get("error_3d_m", pd.Series(dtype=float)))
    e2 = finite_values(errors.get("error_2d_m", pd.Series(dtype=float)))
    denominator = len(errors) if frame_count is None else frame_count
    return {
        "flight": flight_name,
        "method": method,
        "source": source,
        "radar_frame_count": np.nan if frame_count is None else int(frame_count),
        "count": int(e3.size),
        "coverage": safe_divide(float(e3.size), float(denominator)),
        "mean_3d_error_m": stat(e3, "mean"),
        "std_3d_error_m": stat(e3, "std"),
        "rmse_3d_error_m": stat(e3, "rmse"),
        "p95_3d_error_m": stat(e3, "p95"),
        "max_3d_error_m": stat(e3, "max"),
        "mean_2d_error_m": stat(e2, "mean"),
        "std_2d_error_m": stat(e2, "std"),
        "rmse_2d_error_m": stat(e2, "rmse"),
        "p95_2d_error_m": stat(e2, "p95"),
        "max_2d_error_m": stat(e2, "max"),
        "track_switch_count": int(
            errors.get("track_switch", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "mean_selected_cat_prob_uav": stat(
            finite_values(errors.get("cat_prob_uav", pd.Series(dtype=float))), "mean"
        ),
        "p95_association_score": stat(
            finite_values(errors.get("association_score", pd.Series(dtype=float))), "p95"
        ),
        "p95_association_anchor_nis": stat(
            finite_values(errors.get("association_anchor_nis", pd.Series(dtype=float))), "p95"
        ),
    }


def prefixless_summary(summary: dict[str, float]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in summary.items():
        if key in {"count"}:
            out[key] = int(value)
        else:
            out[key] = rounded(value)
    return out


def selected_radar_paths(args: argparse.Namespace, flight_name: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for raw in args.selected_radar_csv:
        if "=" in raw:
            label, path_text = raw.split("=", 1)
            path = Path(path_text)
        else:
            path = Path(raw)
            label = path.parent.name or "selected-radar"
        if path.exists():
            paths.append((label, path))
    for root in args.selected_radar_root:
        if root.is_file():
            paths.append((root.parent.name or "selected-radar", root))
            continue
        candidates = [
            root / flight_name / "selected_radar.csv",
            root / flight_name / "radar_selected.csv",
            root / "selected_radar.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                label = root.name if root.name else candidate.parent.name
                paths.append((label, candidate))
                break
    return paths


def write_plots(
    *,
    output_dir: Path,
    flight_name: str,
    frame_quality: pd.DataFrame,
    oracle: pd.DataFrame,
    rf_errors: pd.DataFrame,
    selected_errors: Sequence[pd.DataFrame],
) -> None:
    write_error_plot(
        output_dir / "errors_over_time.png", flight_name, oracle, rf_errors, selected_errors
    )
    write_candidate_count_plot(
        output_dir / "radar_candidate_counts.png", flight_name, frame_quality
    )
    write_track_id_plot(output_dir / "selected_track_ids.png", flight_name, oracle, selected_errors)


def write_error_plot(
    path: Path,
    flight_name: str,
    oracle: pd.DataFrame,
    rf_errors: pd.DataFrame,
    selected_errors: Sequence[pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_error_series(ax, oracle, "nearest candidate oracle")
    plot_error_series(ax, rf_errors, "RF")
    for frame in selected_errors:
        label = str(frame["method"].iloc[0]) if not frame.empty else "selected radar"
        plot_error_series(ax, frame, label)
    ax.set_title(f"{flight_name}: candidate and measurement error over time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("3D error [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_candidate_count_plot(path: Path, flight_name: str, frame_quality: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if not frame_quality.empty:
        ax.plot(frame_quality["time_s"], frame_quality["candidate_rows"], label="candidate rows")
        ax.plot(
            frame_quality["time_s"],
            frame_quality["unique_track_ids"],
            label="unique track IDs",
        )
    ax.set_title(f"{flight_name}: radar candidate availability")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_track_id_plot(
    path: Path,
    flight_name: str,
    oracle: pd.DataFrame,
    selected_errors: Sequence[pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_track_series(ax, oracle, "nearest candidate oracle")
    for frame in selected_errors:
        label = str(frame["method"].iloc[0]) if not frame.empty else "selected radar"
        plot_track_series(ax, frame, label)
    ax.set_title(f"{flight_name}: selected radar track IDs")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("track ID")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_error_series(ax: plt.Axes, frame: pd.DataFrame, label: str) -> None:
    if frame.empty or "time_s" not in frame.columns or "error_3d_m" not in frame.columns:
        return
    values = pd.to_numeric(frame["error_3d_m"], errors="coerce")
    times = pd.to_numeric(frame["time_s"], errors="coerce")
    finite = np.isfinite(times) & np.isfinite(values)
    if finite.any():
        ax.plot(
            times[finite], values[finite], marker=".", linewidth=1.0, markersize=2.5, label=label
        )


def plot_track_series(ax: plt.Axes, frame: pd.DataFrame, label: str) -> None:
    if frame.empty or "time_s" not in frame.columns or "track_id" not in frame.columns:
        return
    times = pd.to_numeric(frame["time_s"], errors="coerce")
    track_ids = pd.to_numeric(frame["track_id"], errors="coerce")
    finite = np.isfinite(times) & np.isfinite(track_ids)
    if finite.any():
        ax.plot(
            times[finite],
            track_ids[finite],
            marker=".",
            linewidth=0.8,
            markersize=2.5,
            label=label,
        )


def radar_frame_groups(radar: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    if radar.empty:
        return []
    sort_columns = [
        c for c in ("time_s", "frame_index", "track_id", "track_index") if c in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [(key, group.copy()) for key, group in ordered.groupby(group_column, sort=True)]


def track_switch_flags(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "track_id" not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index)
    ids = pd.to_numeric(frame["track_id"], errors="coerce")
    previous = ids.shift(1)
    switches = ids.notna() & previous.notna() & (ids != previous)
    if not switches.empty:
        switches.iloc[0] = False
    return switches.fillna(False)


def inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    lower = float(truth["time_s"].min())
    upper = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= lower) & (frame["time_s"] <= upper)].copy()


def radar_frame_count(radar: pd.DataFrame) -> int:
    if radar.empty:
        return 0
    group_column = "frame_index" if "frame_index" in radar.columns else "time_s"
    return int(radar[group_column].nunique())


def numeric_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def finite_values(values: Iterable[object]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def stat(values: np.ndarray, name: str) -> object:
    if values.size == 0:
        return ""
    if name == "mean":
        return rounded(float(np.mean(values)))
    if name == "std":
        return rounded(float(np.std(values)))
    if name == "rmse":
        return rounded(float(np.sqrt(np.mean(values**2))))
    if name == "p95":
        return rounded(float(np.percentile(values, 95)))
    if name == "max":
        return rounded(float(np.max(values)))
    raise ValueError(f"unknown statistic {name!r}")


def rounded(value: object) -> object:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    return round(out, 3) if np.isfinite(out) else ""


def safe_divide(numerator: float, denominator: float) -> object:
    if denominator <= 0.0:
        return ""
    return rounded(numerator / denominator)


def _matches_discovered(name: str, discovered: Sequence[str]) -> bool:
    return any(name.lower() in flight_name.lower() for flight_name in discovered)


if __name__ == "__main__":
    raise SystemExit(main())
