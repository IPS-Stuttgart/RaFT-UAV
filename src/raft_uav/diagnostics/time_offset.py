"""Timestamp-offset diagnostics for RF/radar measurements."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)

RADAR_SELECTION_MODES = (
    "oracle-nearest-truth",
    "catprob-oracle-nearest",
    "highest-catprob",
    "longest-track",
)
OBJECTIVE_COLUMNS = {
    "mean": "mean_error_m",
    "rmse": "rmse_error_m",
    "p95": "p95_error_m",
    "max": "max_error_m",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-diagnose-time-offset",
        description="sweep RF/radar timestamp offsets against truth telemetry",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--source", choices=["rf", "radar"], required=True)
    parser.add_argument("--tau-min", type=float, default=-15.0)
    parser.add_argument("--tau-max", type=float, default=15.0)
    parser.add_argument("--tau-step", type=float, default=0.1)
    parser.add_argument(
        "--dimensions",
        choices=["auto", "2", "3"],
        default="auto",
        help="error dimensions; auto uses 2D for RF and 3D for radar",
    )
    parser.add_argument(
        "--radar-selection",
        choices=RADAR_SELECTION_MODES,
        default="oracle-nearest-truth",
        help="radar row selection used before computing errors",
    )
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument(
        "--max-truth-time-delta-s",
        type=float,
        default=2.0,
        help="discard comparisons whose shifted time is too far from truth samples",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVE_COLUMNS),
        default="p95",
        help="metric used to choose best_tau_s",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/time-offset"))
    parser.add_argument("--no-plot", action="store_true", help="skip writing offset_sweep.png")
    args = parser.parse_args(argv)

    result = run_time_offset_diagnostic(
        dataset_root=args.dataset_root,
        flight_name=args.flight,
        source=args.source,
        tau_min_s=args.tau_min,
        tau_max_s=args.tau_max,
        tau_step_s=args.tau_step,
        dimensions=args.dimensions,
        radar_selection=args.radar_selection,
        radar_catprob_threshold=args.radar_catprob_threshold,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        objective=args.objective,
        output_dir=args.output_dir,
        write_plot=not args.no_plot,
    )

    print(f"flight={result['flight']}")
    print(f"source={result['source']}")
    print(f"dimensions={result['dimensions']}")
    if result["source"] == "radar":
        print(f"radar_selection={result['radar_selection']}")
    print(f"objective={result['objective']}")
    print(f"best_tau_s={result['best']['tau_s']:.6g}")
    print(f"best_mean_error_m={result['best']['mean_error_m']:.3f}")
    print(f"best_rmse_error_m={result['best']['rmse_error_m']:.3f}")
    print(f"best_p95_error_m={result['best']['p95_error_m']:.3f}")
    print(f"best_coverage={result['best']['coverage']:.3f}")
    print(f"sweep_csv={result['sweep_csv']}")
    print(f"best_json={result['best_json']}")
    if result.get("plot_png"):
        print(f"plot_png={result['plot_png']}")
    return 0


def run_time_offset_diagnostic(
    *,
    dataset_root: Path,
    flight_name: str,
    source: str,
    tau_min_s: float,
    tau_max_s: float,
    tau_step_s: float,
    dimensions: str = "auto",
    radar_selection: str = "oracle-nearest-truth",
    radar_catprob_threshold: float = 0.4,
    max_truth_time_delta_s: float = 2.0,
    objective: str = "p95",
    output_dir: Path = Path("outputs/time-offset"),
    write_plot: bool = True,
) -> dict[str, Any]:
    """Run a timestamp-offset sweep and write CSV/JSON/plot outputs."""

    if source not in {"rf", "radar"}:
        raise ValueError("source must be 'rf' or 'radar'")
    if radar_selection not in RADAR_SELECTION_MODES:
        raise ValueError(f"unknown radar_selection {radar_selection!r}")
    if objective not in OBJECTIVE_COLUMNS:
        raise ValueError(f"unknown objective {objective!r}")
    if tau_step_s <= 0.0:
        raise ValueError("tau_step_s must be positive")
    if tau_max_s < tau_min_s:
        raise ValueError("tau_max_s must be >= tau_min_s")
    if max_truth_time_delta_s <= 0.0:
        raise ValueError("max_truth_time_delta_s must be positive")

    flight = select_flight(Path(dataset_root), flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    truth = truth.sort_values("time_s").reset_index(drop=True)
    taus = offset_grid(tau_min_s, tau_max_s, tau_step_s)
    dims = resolve_dimensions(source, dimensions)

    if source == "rf":
        if flight.rf_csv is None:
            raise FileNotFoundError(f"{flight.name} has no RF CSV file")
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        if rf.empty:
            raise RuntimeError(f"{flight.name} has no RF rows inside the truth window")
        sweep = sweep_positions_against_truth(
            measurement_times_s=rf["time_s"].to_numpy(dtype=float),
            measurement_positions_m=rf[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
            truth=truth,
            taus_s=taus,
            dimensions=dims,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        selection_name = "rf"
    else:
        if flight.radar_json is None:
            raise FileNotFoundError(f"{flight.name} has no radar JSON file")
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
            truth,
        )
        if radar.empty:
            raise RuntimeError(f"{flight.name} has no radar rows inside the truth window")
        sweep = sweep_radar_against_truth(
            radar=radar,
            truth=truth,
            taus_s=taus,
            dimensions=dims,
            selection=radar_selection,
            catprob_threshold=radar_catprob_threshold,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        selection_name = radar_selection

    best = best_offset_row(sweep, objective=objective)
    flight_output = Path(output_dir) / flight.name / f"{source}-{selection_name}"
    flight_output.mkdir(parents=True, exist_ok=True)
    sweep_csv = flight_output / "offset_sweep.csv"
    best_json = flight_output / "best_offset.json"
    plot_png = flight_output / "offset_sweep.png"
    sweep.to_csv(sweep_csv, index=False)

    payload = {
        "flight": flight.name,
        "source": source,
        "radar_selection": radar_selection if source == "radar" else None,
        "dimensions": dims,
        "objective": objective,
        "tau_min_s": float(tau_min_s),
        "tau_max_s": float(tau_max_s),
        "tau_step_s": float(tau_step_s),
        "max_truth_time_delta_s": float(max_truth_time_delta_s),
        "best": _jsonable_row(best),
    }
    best_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    written_plot = write_offset_plot(sweep, objective=objective, output_path=plot_png) if write_plot else None
    return {
        **payload,
        "sweep_csv": str(sweep_csv),
        "best_json": str(best_json),
        "plot_png": None if written_plot is None else str(written_plot),
    }


def sweep_positions_against_truth(
    *,
    measurement_times_s: np.ndarray,
    measurement_positions_m: np.ndarray,
    truth: pd.DataFrame,
    taus_s: Iterable[float],
    dimensions: int,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Sweep offsets for an already-selected sequence of position measurements."""

    times = np.asarray(measurement_times_s, dtype=float).reshape(-1)
    positions = np.asarray(measurement_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < dimensions:
        raise ValueError("measurement_positions_m must be shape (n, >=dimensions)")
    if positions.shape[0] != times.size:
        raise ValueError("measurement times and positions must have the same length")

    rows = []
    for tau_s in taus_s:
        shifted_times = times + float(tau_s)
        truth_positions, mask = truth_positions_at_times(
            truth,
            shifted_times,
            max_delta_s=max_truth_time_delta_s,
        )
        finite = mask & np.isfinite(positions[:, :dimensions]).all(axis=1)
        errors = np.linalg.norm(
            positions[finite, :dimensions] - truth_positions[finite, :dimensions],
            axis=1,
        )
        rows.append(
            summarize_errors(
                tau_s=float(tau_s),
                candidate_count=int(times.size),
                selected_count=int(times.size),
                matched_count=int(errors.size),
                errors_m=errors,
            )
        )
    return pd.DataFrame.from_records(rows)


def sweep_radar_against_truth(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    taus_s: Iterable[float],
    dimensions: int,
    selection: str,
    catprob_threshold: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Sweep offsets while selecting one radar row per frame."""

    groups = radar_frame_groups(radar)
    longest_track_id = _longest_track_id(radar) if selection == "longest-track" else None
    rows = []
    for tau_s in taus_s:
        selected_times, selected_positions = select_radar_rows_for_offset(
            groups=groups,
            truth=truth,
            tau_s=float(tau_s),
            selection=selection,
            catprob_threshold=catprob_threshold,
            longest_track_id=longest_track_id,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        if len(selected_times):
            truth_positions, mask = truth_positions_at_times(
                truth,
                np.asarray(selected_times, dtype=float),
                max_delta_s=max_truth_time_delta_s,
            )
            positions = np.asarray(selected_positions, dtype=float)
            finite = mask & np.isfinite(positions[:, :dimensions]).all(axis=1)
            errors = np.linalg.norm(
                positions[finite, :dimensions] - truth_positions[finite, :dimensions],
                axis=1,
            )
        else:
            errors = np.empty(0, dtype=float)
        rows.append(
            summarize_errors(
                tau_s=float(tau_s),
                candidate_count=len(groups),
                selected_count=len(selected_times),
                matched_count=int(errors.size),
                errors_m=errors,
            )
        )
    return pd.DataFrame.from_records(rows)


def select_radar_rows_for_offset(
    *,
    groups: list[pd.DataFrame],
    truth: pd.DataFrame,
    tau_s: float,
    selection: str,
    catprob_threshold: float,
    longest_track_id: int | None,
    max_truth_time_delta_s: float,
) -> tuple[list[float], list[np.ndarray]]:
    """Select radar positions for one offset value."""

    selected_times: list[float] = []
    selected_positions: list[np.ndarray] = []
    for group in groups:
        shifted_time_s = float(group["time_s"].median()) + float(tau_s)
        truth_position = None
        if selection in {"oracle-nearest-truth", "catprob-oracle-nearest"}:
            truth_position = truth_position_at_time(
                truth,
                shifted_time_s,
                max_delta_s=max_truth_time_delta_s,
            )
            if truth_position is None:
                continue

        if selection == "oracle-nearest-truth":
            selected = nearest_candidate_to_truth(group, truth_position)
        elif selection == "catprob-oracle-nearest":
            selected = nearest_candidate_to_truth(catprob_candidate_pool(group, catprob_threshold), truth_position)
        elif selection == "highest-catprob":
            selected = highest_catprob_candidate(group)
        elif selection == "longest-track":
            if longest_track_id is None or "track_id" not in group.columns:
                selected = None
            else:
                track_rows = group.loc[pd.to_numeric(group["track_id"], errors="coerce") == longest_track_id]
                selected = highest_catprob_candidate(track_rows)
        else:
            raise ValueError(f"unknown radar selection {selection!r}")

        if selected is None:
            continue
        selected_times.append(float(shifted_time_s))
        selected_positions.append(selected[["east_m", "north_m", "up_m"]].to_numpy(dtype=float))
    return selected_times, selected_positions


def offset_grid(tau_min_s: float, tau_max_s: float, tau_step_s: float) -> np.ndarray:
    """Return an inclusive floating-point offset grid."""

    count = int(np.floor((float(tau_max_s) - float(tau_min_s)) / float(tau_step_s) + 1e-9))
    values = float(tau_min_s) + np.arange(count + 1, dtype=float) * float(tau_step_s)
    if values.size == 0 or values[-1] < float(tau_max_s) - 1e-9:
        values = np.append(values, float(tau_max_s))
    return values


def truth_positions_at_times(
    truth: pd.DataFrame,
    times_s: np.ndarray,
    *,
    max_delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate truth ENU positions and return validity mask."""

    query = np.asarray(times_s, dtype=float).reshape(-1)
    truth_numeric = truth[["time_s", "east_m", "north_m", "up_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite_truth = np.isfinite(truth_numeric.to_numpy(dtype=float)).all(axis=1)
    truth_numeric = (
        truth_numeric.loc[finite_truth]
        .groupby("time_s", as_index=False)
        .median()
        .sort_values("time_s")
    )
    truth_times = truth_numeric["time_s"].to_numpy(dtype=float)
    truth_positions = truth_numeric[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    if truth_times.size == 0:
        return np.full((query.size, 3), np.nan), np.zeros(query.size, dtype=bool)

    interpolated = np.column_stack(
        [np.interp(query, truth_times, truth_positions[:, axis]) for axis in range(3)]
    )
    insertion = np.searchsorted(truth_times, query)
    right = np.clip(insertion, 0, truth_times.size - 1)
    left = np.clip(insertion - 1, 0, truth_times.size - 1)
    nearest_delta = np.minimum(np.abs(truth_times[right] - query), np.abs(truth_times[left] - query))
    valid = (
        np.isfinite(query)
        & (query >= truth_times[0])
        & (query <= truth_times[-1])
        & (nearest_delta <= float(max_delta_s))
    )
    return interpolated, valid


def truth_position_at_time(truth: pd.DataFrame, time_s: float, *, max_delta_s: float) -> np.ndarray | None:
    positions, mask = truth_positions_at_times(
        truth,
        np.array([float(time_s)]),
        max_delta_s=max_delta_s,
    )
    return positions[0] if bool(mask[0]) else None


def summarize_errors(
    *,
    tau_s: float,
    candidate_count: int,
    selected_count: int,
    matched_count: int,
    errors_m: np.ndarray,
) -> dict[str, float | int]:
    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    summary: dict[str, float | int] = {
        "tau_s": float(tau_s),
        "candidate_count": int(candidate_count),
        "selected_count": int(selected_count),
        "matched_count": int(matched_count),
        "coverage": _safe_ratio(matched_count, candidate_count),
        "selected_coverage": _safe_ratio(matched_count, selected_count),
    }
    if errors.size == 0:
        summary.update(
            {
                "mean_error_m": float("nan"),
                "std_error_m": float("nan"),
                "rmse_error_m": float("nan"),
                "p50_error_m": float("nan"),
                "p95_error_m": float("nan"),
                "max_error_m": float("nan"),
            }
        )
        return summary
    summary.update(
        {
            "mean_error_m": float(np.mean(errors)),
            "std_error_m": float(np.std(errors)),
            "rmse_error_m": float(np.sqrt(np.mean(errors**2))),
            "p50_error_m": float(np.percentile(errors, 50.0)),
            "p95_error_m": float(np.percentile(errors, 95.0)),
            "max_error_m": float(np.max(errors)),
        }
    )
    return summary


def best_offset_row(sweep: pd.DataFrame, *, objective: str) -> pd.Series:
    column = OBJECTIVE_COLUMNS[objective]
    values = pd.to_numeric(sweep[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not finite.any():
        raise RuntimeError(f"no finite {column} values in offset sweep")
    finite_indices = np.flatnonzero(finite)
    best_position = finite_indices[int(np.argmin(values.to_numpy(dtype=float)[finite]))]
    return sweep.iloc[int(best_position)]


def write_offset_plot(sweep: pd.DataFrame, *, objective: str, output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    column = OBJECTIVE_COLUMNS[objective]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["tau_s"], sweep[column], marker=".")
    ax.set_xlabel("Applied sensor time shift tau [s]")
    ax.set_ylabel(column.replace("_", " "))
    ax.set_title("Timestamp-offset sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty:
        return []
    sort_columns = [column for column in ("time_s", "frame_index", "track_id", "track_index") if column in radar.columns]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    group_column = "frame_index" if "frame_index" in ordered.columns else "time_s"
    return [group.copy() for _, group in ordered.groupby(group_column, sort=True)]


def nearest_candidate_to_truth(candidates: pd.DataFrame, truth_position: np.ndarray | None) -> pd.Series | None:
    if truth_position is None or candidates.empty:
        return None
    candidate_xyz = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors = np.linalg.norm(candidate_xyz - truth_position.reshape(1, 3), axis=1)
    if not np.isfinite(errors).any():
        return None
    return candidates.iloc[int(np.nanargmin(errors))].copy()


def highest_catprob_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    if "cat_prob_uav" not in candidates.columns:
        return candidates.iloc[0].copy()
    scores = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(-np.inf)
    return candidates.loc[scores.idxmax()].copy()


def catprob_candidate_pool(candidates: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if candidates.empty or "cat_prob_uav" not in candidates.columns:
        return candidates
    scores = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce")
    keep = scores >= float(threshold)
    return candidates.loc[keep].copy() if keep.any() else candidates


def resolve_dimensions(source: str, dimensions: str) -> int:
    if dimensions == "auto":
        return 2 if source == "rf" else 3
    return int(dimensions)


def _longest_track_id(radar: pd.DataFrame) -> int | None:
    if "track_id" not in radar.columns or radar.empty:
        return None
    track_ids = pd.to_numeric(radar["track_id"], errors="coerce").dropna()
    if track_ids.empty:
        return None
    counts = track_ids.astype(int).value_counts()
    return int(counts.idxmax())


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    start = float(truth["time_s"].min())
    end = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= start) & (frame["time_s"] <= end)].copy()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) > 0 else float("nan")


def _jsonable_row(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            out[str(key)] = None
        else:
            out[str(key)] = value
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
