"""Local evaluator helpers for UG2+/MMUAD-style trajectory exports.

This module intentionally implements transparent local metrics, not the closed
Codabench runtime.  It validates ``mmaud_results.csv``-style files and can
evaluate either a nearest-time development diagnostic or the public Track 5
timestamp-aligned MSE/classification quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipInfo
from zipfile import ZipFile

import json
import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    TruthFrame,
    load_jsonable,
    normalize_time_column_aliases,
    normalize_truth_columns,
)
from raft_uav.mmuad import _submission_impl
from raft_uav.mmuad.submission import (
    OFFICIAL_UG2_RESULT_COLUMNS,
    UG2_RESULT_COLUMNS,
    load_sequence_class_map,
    parse_official_classification_cell,
    parse_official_position_cell,
    parse_official_sequence_cell,
    parse_official_timestamp_cell,
)

_TRUTH_TYPE_COLUMNS = (
    "uav_type",
    "class_name",
    "class",
    "label",
    "category",
    "classification",
    "class_id",
    "uav_type_id",
    "type_id",
)


@dataclass(frozen=True)
class ResultsFrame:
    """Validated UG2-style result rows."""

    rows: pd.DataFrame


def load_mmaud_results_csv(path: Path) -> ResultsFrame:
    """Load and validate a ``mmaud_results.csv``-style file."""

    frame = pd.read_csv(path)
    return ResultsFrame(validate_mmaud_results_frame(frame))


def load_mmaud_results_file(path: Path) -> ResultsFrame:
    """Load result rows from a CSV file or a Codabench-style ZIP archive."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        return load_mmaud_results_zip(path)
    return load_mmaud_results_csv(path)


def load_evaluation_truth_file(path: Path) -> TruthFrame:
    """Load normalized or official Track 5 truth rows for local evaluation."""

    from raft_uav.mmuad.io import load_truth_file

    try:
        return load_truth_file(path)
    except ValueError:
        try:
            rows = _load_official_track5_truth_file(path)
        except Exception as official_error:
            raise ValueError(
                "evaluation truth file must be a normalized truth table or "
                "an official Track 5 CSV/ZIP with Sequence, Timestamp, "
                "Position, and Classification"
            ) from official_error
    frame = TruthFrame(rows)
    frame.validate()
    return frame


def load_mmaud_results_zip(
    path: Path,
    *,
    member_name: str = "mmaud_results.csv",
) -> ResultsFrame:
    """Load and validate result rows from a ZIP archive.

    Public Track 5 upload readiness still requires root ``mmaud_results.csv``.
    Evaluation is more permissive so local diagnostics can score archives before
    they are normalized: root ``mmaud_results.csv``, one nested
    ``mmaud_results.csv``, or one unambiguous CSV member are accepted.
    """

    path = Path(path)
    frame = _read_results_zip_csv(path, member_name=member_name)
    return ResultsFrame(validate_mmaud_results_frame(frame))


def validate_mmaud_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized result frame or raise with actionable errors."""

    if _has_official_track5_columns(frame):
        frame = _official_track5_results_to_local_frame(frame)

    rename = {
        "time_s": "timestamp",
        "t": "timestamp",
        "x_m": "x",
        "y_m": "y",
        "z_m": "z",
        "class_name": "uav_type",
        "label": "uav_type",
        "confidence": "score",
    }
    rows = normalize_time_column_aliases(frame, target="timestamp")
    rows = rows.rename(
        columns={
            key: value
            for key, value in rename.items()
            if key in rows.columns and value not in rows.columns
        }
    ).copy()
    missing = set(UG2_RESULT_COLUMNS).difference(rows.columns)
    if missing:
        raise ValueError(f"mmaud_results rows missing columns: {sorted(missing)}")
    rows = rows[list(UG2_RESULT_COLUMNS)].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["uav_type"] = rows["uav_type"].astype(str)
    for col in ("timestamp", "x", "y", "z", "score"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    finite = np.isfinite(rows[["timestamp", "x", "y", "z", "score"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    if rows.empty:
        raise ValueError("mmaud_results contains no finite trajectory rows")
    return rows.sort_values(["sequence_id", "timestamp"]).reset_index(drop=True)


def _official_track5_column_map(frame: pd.DataFrame) -> dict[str, Any]:
    """Map normalized official Track 5 column names to their original labels."""

    return {str(column).strip().lower(): column for column in frame.columns}


def _has_official_track5_columns(frame: pd.DataFrame) -> bool:
    lower = set(_official_track5_column_map(frame))
    return {column.lower() for column in OFFICIAL_UG2_RESULT_COLUMNS}.issubset(lower)


def _official_track5_results_to_local_frame(frame: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = _official_track5_column_map(frame)
    sequence_col = lower_to_original["sequence"]
    timestamp_col = lower_to_original["timestamp"]
    position_col = lower_to_original["position"]
    classification_col = lower_to_original["classification"]
    sequences = [parse_official_sequence_cell(value) for value in frame[sequence_col]]
    timestamps = [parse_official_timestamp_cell(value) for value in frame[timestamp_col]]
    positions = [parse_official_position_cell(value) for value in frame[position_col]]
    classifications = [
        parse_official_classification_cell(value)
        for value in frame[classification_col]
    ]
    xyz = pd.DataFrame(positions, columns=["x", "y", "z"], index=frame.index)
    return pd.DataFrame(
        {
            "sequence_id": sequences,
            "timestamp": timestamps,
            "x": xyz["x"],
            "y": xyz["y"],
            "z": xyz["z"],
            "uav_type": [str(value) for value in classifications],
            "score": 1.0,
        }
    )


def _load_official_track5_truth_file(path: Path) -> pd.DataFrame:
    frame, _ = _submission_impl._read_official_track5_results_input(Path(path))
    normalizer = getattr(
        _submission_impl,
        "_raft_uav_original_normalize_official_track5_results_frame",
        _submission_impl.normalize_official_track5_results_frame,
    )
    frame = normalizer(frame)
    return _official_track5_truth_to_rows(frame)


def _read_results_zip_csv(path: Path, *, member_name: str) -> pd.DataFrame:
    with ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        selected = _select_results_zip_member(infos, member_name=member_name)
        with archive.open(selected) as handle:
            return pd.read_csv(BytesIO(handle.read()))


def _select_results_zip_member(
    infos: list[ZipInfo],
    *,
    member_name: str,
) -> ZipInfo:
    root_results = [
        info
        for info in infos
        if _normalized_zip_member_name(info.filename) == member_name
    ]
    basename_results = [
        info
        for info in infos
        if Path(_normalized_zip_member_name(info.filename)).name == member_name
    ]
    csv_members = [
        info
        for info in infos
        if Path(_normalized_zip_member_name(info.filename)).suffix.lower() == ".csv"
    ]
    if len(root_results) > 1:
        raise ValueError(
            f"results ZIP has duplicate root {member_name!r} members"
        )
    if root_results:
        return root_results[0]
    if len(basename_results) == 1:
        return basename_results[0]
    if len(csv_members) == 1:
        return csv_members[0]
    names = [info.filename for info in infos]
    raise ValueError(
        "results ZIP must contain an unambiguous CSV: root "
        f"{member_name}, one nested {member_name}, or one CSV member; members={names}"
    )


def _normalized_zip_member_name(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("/")


def _official_track5_truth_to_rows(frame: pd.DataFrame) -> pd.DataFrame:
    local = _official_track5_results_to_local_frame(frame)
    rows = local.rename(
        columns={
            "timestamp": "time_s",
            "x": "x_m",
            "y": "y_m",
            "z": "z_m",
            "uav_type": "class_name",
        }
    )
    return normalize_truth_columns(rows)


def evaluate_mmaud_results(
    results: ResultsFrame | pd.DataFrame,
    truth: TruthFrame | pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
    metric_protocol: str = "nearest-time",
    timestamp_tolerance_s: float = 1.0e-6,
    class_map_csv: Path | None = None,
    class_map_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate result rows against normalized truth.

    ``nearest-time`` is an ADE/FDE-style development diagnostic.  ``public-track5``
    aligns predictions to the truth/template timestamps required by the public
    UG2+ Track 5 submission instructions and reports the public MSE and
    classification-accuracy quantities.  Neither mode claims closed Codabench
    runtime equivalence.
    """

    result_rows = (
        results.rows if isinstance(results, ResultsFrame) else validate_mmaud_results_frame(results)
    )
    truth_rows = truth.rows if isinstance(truth, TruthFrame) else normalize_truth_columns(truth)
    if result_rows.empty or truth_rows.empty:
        return {"num_matches": 0, "rmse_m": np.nan, "mean_error_m": np.nan, "max_error_m": np.nan}
    if class_map_path is not None:
        if class_map_csv is not None and Path(class_map_csv) != Path(class_map_path):
            raise ValueError("provide only one of class_map_csv or class_map_path")
        class_map_csv = class_map_path
    class_map = load_sequence_class_map(class_map_csv) if class_map_csv is not None else None
    if metric_protocol == "nearest-time":
        return _evaluate_nearest_time(result_rows, truth_rows, max_time_delta_s=max_time_delta_s)
    if metric_protocol == "public-track5":
        return _evaluate_public_track5(
            result_rows,
            truth_rows,
            timestamp_tolerance_s=timestamp_tolerance_s,
            class_map=class_map,
        )
    raise ValueError("metric_protocol must be 'nearest-time' or 'public-track5'")


def _evaluate_nearest_time(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> dict[str, Any]:
    matches: list[float] = []
    for seq, result_seq in results.groupby("sequence_id"):
        truth_seq = truth.loc[truth["sequence_id"].astype(str) == str(seq)].sort_values("time_s")
        if truth_seq.empty:
            continue
        truth_times = truth_seq["time_s"].to_numpy(float)
        truth_xyz = truth_seq[["x_m", "y_m", "z_m"]].to_numpy(float)
        for _, row in result_seq.sort_values("timestamp").iterrows():
            timestamp = float(row["timestamp"])
            idx = int(np.argmin(np.abs(truth_times - timestamp)))
            dt = float(abs(truth_times[idx] - timestamp))
            if dt > max_time_delta_s:
                continue
            pred = row[["x", "y", "z"]].to_numpy(float)
            matches.append(float(np.linalg.norm(pred - truth_xyz[idx])))
    if not matches:
        return {"num_matches": 0, "rmse_m": np.nan, "mean_error_m": np.nan, "max_error_m": np.nan}
    arr = np.asarray(matches, dtype=float)
    return {
        "num_matches": int(len(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr**2))),
        "mean_error_m": float(np.mean(arr)),
        "max_error_m": float(np.max(arr)),
    }


def _evaluate_public_track5(
    results: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    timestamp_tolerance_s: float,
    class_map: dict[str, str] | None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for _, truth_row in truth.iterrows():
        seq = str(truth_row["sequence_id"])
        timestamp = float(truth_row["time_s"])
        seq_results = results.loc[results["sequence_id"].astype(str) == seq].copy()
        if seq_results.empty:
            continue
        dt = np.abs(pd.to_numeric(seq_results["timestamp"], errors="coerce").to_numpy(float) - timestamp)
        if not np.isfinite(dt).any():
            continue
        idx = int(np.nanargmin(dt))
        if float(dt[idx]) > float(timestamp_tolerance_s):
            continue
        row = seq_results.iloc[idx]
        pred = row[["x", "y", "z"]].to_numpy(float)
        truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
        error = float(np.linalg.norm(pred - truth_xyz))
        truth_class = _truth_class_for_row(truth_row, seq, class_map)
        predicted_class = str(row["uav_type"])
        class_correct = truth_class is not None and predicted_class == truth_class
        matches.append(
            {
                "sequence_id": seq,
                "time_s": timestamp,
                "error_m": error,
                "squared_error_m2": error**2,
                "truth_class": truth_class,
                "predicted_class": predicted_class,
                "class_correct": bool(class_correct),
            }
        )
    if not matches:
        return {
            "metric_protocol": "public-track5",
            "num_matches": 0,
            "pose_mse_m2": np.nan,
            "pose_rmse_m": np.nan,
            "pose_mean_m": np.nan,
            "pose_max_m": np.nan,
            "classification_accuracy": np.nan,
        }
    match_rows = pd.DataFrame.from_records(matches)
    return {
        "metric_protocol": "public-track5",
        "num_matches": int(len(match_rows)),
        "pose_mse_m2": float(match_rows["squared_error_m2"].mean()),
        "pose_rmse_m": float(np.sqrt(match_rows["squared_error_m2"].mean())),
        "pose_mean_m": float(match_rows["error_m"].mean()),
        "pose_max_m": float(match_rows["error_m"].max()),
        "classification_accuracy": float(match_rows["class_correct"].mean()),
    }


def _truth_class_for_row(
    truth_row: pd.Series,
    sequence_id: str,
    class_map: dict[str, str] | None,
) -> str | None:
    if class_map is not None and sequence_id in class_map:
        return str(class_map[sequence_id])
    for column in _TRUTH_TYPE_COLUMNS:
        if column in truth_row.index and not pd.isna(truth_row[column]):
            return str(truth_row[column])
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate UG2/MMUAD trajectory results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--max-time-delta-s", type=float, default=0.5)
    parser.add_argument(
        "--metric-protocol",
        choices=("nearest-time", "public-track5"),
        default="nearest-time",
    )
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--class-map-csv", type=Path)
    args = parser.parse_args(argv)

    result_frame = load_mmaud_results_file(args.results)
    truth_frame = load_evaluation_truth_file(args.truth)
    metrics = evaluate_mmaud_results(
        result_frame,
        truth_frame,
        max_time_delta_s=args.max_time_delta_s,
        metric_protocol=args.metric_protocol,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
        class_map_csv=args.class_map_csv,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
