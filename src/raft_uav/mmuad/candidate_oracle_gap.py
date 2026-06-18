"""Nearest-candidate oracle-gap diagnostics for MMUAD tracking runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file, merge_candidate_frames
from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export


OUTPUT_NAME = "mmuad_candidate_oracle_gap.csv"


def build_candidate_oracle_gap(
    candidates: CandidateFrame | pd.DataFrame,
    selected: CandidateFrame | pd.DataFrame,
    truth: TruthFrame | pd.DataFrame,
    *,
    max_time_delta_s: float | None = 0.5,
) -> pd.DataFrame:
    """Compare tracker-selected candidates with nearest raw candidates to truth.

    Rows are emitted for every truth timestamp and every candidate source/sensor
    present in the sequence.  The nearest raw candidate is source-specific,
    while the selected candidate is the tracker-selected row nearest in time at
    that timestamp.  Positive ``candidate_regret_m`` means the raw candidate for
    that sensor was closer to truth than the tracker-selected candidate.
    """

    candidate_rows = _finite_candidate_rows(_as_candidate_rows(candidates))
    selected_rows = _finite_candidate_rows(_as_candidate_rows(selected, default_source="selected"))
    truth_rows = _finite_truth_rows(_as_truth_rows(truth))
    if truth_rows.empty:
        return pd.DataFrame(columns=_oracle_gap_columns())
    candidate_rows = _with_source_norm(candidate_rows)
    selected_rows = _with_source_norm(selected_rows)
    records: list[dict[str, Any]] = []
    sequence_ids = sorted(
        set(truth_rows["sequence_id"].astype(str))
        | set(candidate_rows["sequence_id"].astype(str))
        | set(selected_rows["sequence_id"].astype(str))
    )
    for sequence_id in sequence_ids:
        sequence_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == sequence_id]
        if sequence_truth.empty:
            continue
        sequence_candidates = candidate_rows.loc[
            candidate_rows["sequence_id"].astype(str) == sequence_id
        ]
        sequence_selected = selected_rows.loc[selected_rows["sequence_id"].astype(str) == sequence_id]
        sensors = sorted(
            set(sequence_candidates["_source_norm"].astype(str))
            | set(sequence_selected["_source_norm"].astype(str))
        )
        if not sensors:
            sensors = ["candidate"]
        candidates_by_sensor = {
            sensor: group.sort_values("time_s").reset_index(drop=True)
            for sensor, group in sequence_candidates.groupby("_source_norm", sort=True)
        }
        selected_sorted = sequence_selected.sort_values("time_s").reset_index(drop=True)
        for _, truth_row in sequence_truth.sort_values("time_s").iterrows():
            selected_at_time = _nearest_time_rows(
                selected_sorted,
                float(truth_row["time_s"]),
                max_time_delta_s=max_time_delta_s,
            )
            selected_best = _best_candidate_to_truth(selected_at_time, truth_row)
            for sensor in sensors:
                raw_at_time = _nearest_time_rows(
                    candidates_by_sensor.get(sensor, sequence_candidates.iloc[0:0]),
                    float(truth_row["time_s"]),
                    max_time_delta_s=max_time_delta_s,
                )
                nearest_best = _best_candidate_to_truth(raw_at_time, truth_row)
                records.append(
                    _gap_record(
                        truth_row,
                        sensor=sensor,
                        selected=selected_best,
                        nearest=nearest_best,
                        candidate_count_at_nearest_time=len(raw_at_time),
                    )
                )
    return pd.DataFrame.from_records(records, columns=_oracle_gap_columns())


def build_candidate_oracle_gap_from_sequence_root(
    sequence_root: Path | None,
    truth_path: Path,
    selected_tracklets: Path,
    *,
    sequence_glob: str = "*",
    candidate_csvs: tuple[Path, ...] = (),
    apply_calibration: bool = True,
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
    max_time_delta_s: float | None = 0.5,
) -> pd.DataFrame:
    """Load an MMUAD sequence root and write per-sensor oracle-gap rows."""

    truth = load_evaluation_truth_file(Path(truth_path))
    selected = load_candidate_file(Path(selected_tracklets), source="selected")
    candidates = _load_candidates(
        sequence_root,
        sequence_glob=sequence_glob,
        candidate_csvs=candidate_csvs,
        apply_calibration=apply_calibration,
        voxel_size_m=voxel_size_m,
        min_cluster_points=min_cluster_points,
    )
    return build_candidate_oracle_gap(
        candidates,
        selected,
        truth,
        max_time_delta_s=max_time_delta_s,
    )


def write_candidate_oracle_gap(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``mmuad_candidate_oracle_gap.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _load_candidates(
    sequence_root: Path | None,
    *,
    sequence_glob: str,
    candidate_csvs: tuple[Path, ...],
    apply_calibration: bool,
    voxel_size_m: float,
    min_cluster_points: int,
) -> CandidateFrame:
    frames: list[CandidateFrame] = []
    for path in candidate_csvs:
        frames.append(load_candidate_file(Path(path)))
    if sequence_root is not None:
        sequences = discover_sequence_paths(Path(sequence_root), sequence_glob=sequence_glob)
        for paths in sequences:
            try:
                candidates, _, _ = load_sequence_export(
                    paths,
                    apply_calibration=apply_calibration,
                    voxel_size_m=voxel_size_m,
                    min_cluster_points=min_cluster_points,
                )
            except Exception:
                continue
            if not candidates.rows.empty:
                frames.append(candidates)
    if not frames:
        return CandidateFrame(normalize_candidate_columns(pd.DataFrame()))
    return merge_candidate_frames(frames)


def _as_candidate_rows(
    frame: CandidateFrame | pd.DataFrame,
    *,
    default_source: str = "candidate",
) -> pd.DataFrame:
    if isinstance(frame, CandidateFrame):
        return frame.rows.copy()
    return normalize_candidate_columns(pd.DataFrame(frame), default_source=default_source)


def _as_truth_rows(frame: TruthFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame, TruthFrame):
        return frame.rows.copy()
    return normalize_truth_columns(pd.DataFrame(frame))


def _with_source_norm(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if rows.empty:
        rows["_source_norm"] = pd.Series(dtype=object)
        return rows
    rows["_source_norm"] = rows["source"].map(_source_norm)
    return rows


def _source_norm(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    return text or "candidate"


def _finite_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = truth.copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _finite_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "time_s",
                "source",
                "track_id",
                "x_m",
                "y_m",
                "z_m",
                "confidence",
            ]
        )
    if "track_id" not in rows.columns:
        rows["track_id"] = np.nan
    if "confidence" not in rows.columns:
        rows["confidence"] = np.nan
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m", "confidence"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s", "source"]).reset_index(drop=True)


def _nearest_time_rows(
    rows: pd.DataFrame,
    time_s: float,
    *,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    times = pd.to_numeric(rows["time_s"], errors="coerce")
    finite = np.isfinite(times.to_numpy(float))
    if not finite.any():
        return rows.iloc[0:0].copy()
    deltas = (times - float(time_s)).abs()
    best_delta = float(deltas.loc[finite].min())
    if max_time_delta_s is not None and best_delta > float(max_time_delta_s):
        return rows.iloc[0:0].copy()
    return rows.loc[finite & (np.abs(deltas - best_delta) <= 1.0e-9)].copy()


def _best_candidate_to_truth(rows: pd.DataFrame, truth_row: pd.Series) -> pd.Series | None:
    if rows.empty:
        return None
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    candidate_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
    if not np.isfinite(distances).any():
        return None
    return rows.iloc[int(np.nanargmin(distances))]


def _gap_record(
    truth_row: pd.Series,
    *,
    sensor: str,
    selected: pd.Series | None,
    nearest: pd.Series | None,
    candidate_count_at_nearest_time: int,
) -> dict[str, Any]:
    selected_error = _candidate_error(selected, truth_row)
    nearest_error = _candidate_error(nearest, truth_row)
    regret = (
        float(selected_error - nearest_error)
        if np.isfinite(selected_error) and np.isfinite(nearest_error)
        else np.nan
    )
    selected_sensor = _source_norm(selected.get("source")) if selected is not None else ""
    return {
        "sequence": str(truth_row["sequence_id"]),
        "sequence_id": str(truth_row["sequence_id"]),
        "time_s": float(truth_row["time_s"]),
        "sensor": sensor,
        "truth_x_m": float(truth_row["x_m"]),
        "truth_y_m": float(truth_row["y_m"]),
        "truth_z_m": float(truth_row["z_m"]),
        "nearest_candidate_found": nearest is not None,
        "nearest_candidate_time_s": _candidate_value(nearest, "time_s"),
        "nearest_candidate_time_delta_s": _candidate_time_delta(nearest, truth_row),
        "nearest_candidate_source": _candidate_text(nearest, "source"),
        "nearest_candidate_track_id": _candidate_text(nearest, "track_id"),
        "nearest_candidate_confidence": _candidate_value(nearest, "confidence"),
        "nearest_candidate_x_m": _candidate_value(nearest, "x_m"),
        "nearest_candidate_y_m": _candidate_value(nearest, "y_m"),
        "nearest_candidate_z_m": _candidate_value(nearest, "z_m"),
        "candidate_count_at_nearest_time": int(candidate_count_at_nearest_time),
        "selected_candidate_found": selected is not None,
        "selected_candidate_time_s": _candidate_value(selected, "time_s"),
        "selected_candidate_time_delta_s": _candidate_time_delta(selected, truth_row),
        "selected_candidate_source": _candidate_text(selected, "source"),
        "selected_candidate_track_id": _candidate_text(selected, "track_id"),
        "selected_candidate_confidence": _candidate_value(selected, "confidence"),
        "selected_candidate_x_m": _candidate_value(selected, "x_m"),
        "selected_candidate_y_m": _candidate_value(selected, "y_m"),
        "selected_candidate_z_m": _candidate_value(selected, "z_m"),
        "selected_source_matches_sensor": bool(selected_sensor == sensor) if selected is not None else False,
        "selected_minus_truth_error_m": selected_error,
        "nearest_minus_truth_error_m": nearest_error,
        "candidate_regret_m": regret,
    }


def _candidate_error(row: pd.Series | None, truth_row: pd.Series) -> float:
    if row is None:
        return np.nan
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    candidate_xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
    return float(np.linalg.norm(candidate_xyz - truth_xyz))


def _candidate_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def _candidate_time_delta(row: pd.Series | None, truth_row: pd.Series) -> float:
    value = _candidate_value(row, "time_s")
    if not np.isfinite(value):
        return np.nan
    return float(value - float(truth_row["time_s"]))


def _candidate_text(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column])


def _oracle_gap_columns() -> list[str]:
    return [
        "sequence",
        "sequence_id",
        "time_s",
        "sensor",
        "truth_x_m",
        "truth_y_m",
        "truth_z_m",
        "nearest_candidate_found",
        "nearest_candidate_time_s",
        "nearest_candidate_time_delta_s",
        "nearest_candidate_source",
        "nearest_candidate_track_id",
        "nearest_candidate_confidence",
        "nearest_candidate_x_m",
        "nearest_candidate_y_m",
        "nearest_candidate_z_m",
        "candidate_count_at_nearest_time",
        "selected_candidate_found",
        "selected_candidate_time_s",
        "selected_candidate_time_delta_s",
        "selected_candidate_source",
        "selected_candidate_track_id",
        "selected_candidate_confidence",
        "selected_candidate_x_m",
        "selected_candidate_y_m",
        "selected_candidate_z_m",
        "selected_source_matches_sensor",
        "selected_minus_truth_error_m",
        "nearest_minus_truth_error_m",
        "candidate_regret_m",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-oracle-gap",
        description="write mmuad_candidate_oracle_gap.csv for MMUAD tracker diagnostics",
    )
    parser.add_argument(
        "sequence_root",
        nargs="?",
        type=Path,
        help="MMUAD sequence root; optional when --candidate-csv is supplied",
    )
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument("--selected-tracklets", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        action="append",
        default=[],
        help="optional normalized raw candidate CSV; may be repeated",
    )
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--max-time-delta-s", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--apply-calibration", dest="apply_calibration", action="store_true")
    calibration.add_argument("--no-apply-calibration", dest="apply_calibration", action="store_false")
    parser.set_defaults(apply_calibration=True)
    args = parser.parse_args(argv)

    if args.sequence_root is None and not args.candidate_csv:
        parser.error("provide sequence_root or at least one --candidate-csv")
    output_csv = args.output_csv or (args.output_dir / OUTPUT_NAME)
    frame = build_candidate_oracle_gap_from_sequence_root(
        args.sequence_root,
        args.truth_file,
        args.selected_tracklets,
        sequence_glob=args.sequence_glob,
        candidate_csvs=tuple(args.candidate_csv),
        apply_calibration=bool(args.apply_calibration),
        voxel_size_m=float(args.voxel_size_m),
        min_cluster_points=int(args.min_cluster_points),
        max_time_delta_s=float(args.max_time_delta_s),
    )
    path = write_candidate_oracle_gap(frame, output_csv)
    print("mmuad_candidate_oracle_gap=ok")
    print(f"output_csv={path}")
    print(f"row_count={len(frame)}")
    if not frame.empty:
        selected = frame["selected_candidate_found"].astype(bool)
        nearest = frame["nearest_candidate_found"].astype(bool)
        finite_regret = pd.to_numeric(frame["candidate_regret_m"], errors="coerce")
        finite_regret = finite_regret[np.isfinite(finite_regret)]
        print(f"selected_found_fraction={float(selected.mean())}")
        print(f"nearest_found_fraction={float(nearest.mean())}")
        if not finite_regret.empty:
            print(f"mean_candidate_regret_m={float(finite_regret.mean())}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
