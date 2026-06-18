"""Lightweight UAV type helpers for MMUAD/UG2-style submissions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import numpy as np
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.submission import (
    load_official_track5_results_frame,
    load_sequence_class_map,
)


UNKNOWN_LABELS = {"", "unknown", "nan", "none", "uav", "drone"}
CLASSIFIER_METADATA_COLUMNS = {"sequence_id", "uav_type"}
SEQUENCE_ID_ALIASES = ("sequence_id", "Sequence", "sequence", "seq", "scene_id", "id")
TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "timestamp_s", "t", "time")
SOURCE_ALIASES = ("source", "sensor", "modality")
TRACK_ID_ALIASES = ("track_id", "track", "object_id", "cluster_id", "id")
X_ALIASES = ("state_x_m", "x_m", "x", "east_m", "position_x", "center_x")
Y_ALIASES = ("state_y_m", "y_m", "y", "north_m", "position_y", "center_y")
Z_ALIASES = ("state_z_m", "z_m", "z", "up_m", "position_z", "center_z")
VX_ALIASES = ("v_x_mps", "vx_mps", "vx", "velocity_x")
VY_ALIASES = ("v_y_mps", "vy_mps", "vy", "velocity_y")
VZ_ALIASES = ("v_z_mps", "vz_mps", "vz", "velocity_z")
CONFIDENCE_ALIASES = ("confidence", "score", "probability")
STD_XY_ALIASES = ("std_xy_m", "xy_std_m", "std_m", "sigma_xy_m")
STD_Z_ALIASES = ("std_z_m", "z_std_m", "sigma_z_m")
CLUSTER_POINT_COUNT_ALIASES = (
    "cluster_point_count",
    "point_count",
    "points",
    "num_points",
    "n_points",
    "return_count",
)
CLUSTER_EXTENT_X_ALIASES = (
    "cluster_extent_x_m",
    "bbox_extent_x_m",
    "extent_x_m",
    "bbox_size_x_m",
    "size_x_m",
    "width_m",
)
CLUSTER_EXTENT_Y_ALIASES = (
    "cluster_extent_y_m",
    "bbox_extent_y_m",
    "extent_y_m",
    "bbox_size_y_m",
    "size_y_m",
    "depth_m",
)
CLUSTER_EXTENT_Z_ALIASES = (
    "cluster_extent_z_m",
    "bbox_extent_z_m",
    "extent_z_m",
    "bbox_size_z_m",
    "size_z_m",
    "height_m",
)
CLUSTER_EXTENT_XY_ALIASES = ("cluster_extent_xy_m", "bbox_extent_xy_m", "extent_xy_m")
CLUSTER_EXTENT_3D_ALIASES = (
    "cluster_extent_3d_m",
    "bbox_extent_3d_m",
    "extent_3d_m",
    "cluster_size_m",
)
CLUSTER_DENSITY_ALIASES = (
    "cluster_density_points_per_m3",
    "density_points_per_m3",
    "point_density",
)
EMPTY_FRAME_ALIASES = (
    "empty_frame",
    "is_empty",
    "frame_empty",
    "radar_empty",
    "empty_radar_frame",
)
ROW_LEVEL_ALIASES = (
    TIME_ALIASES
    + X_ALIASES
    + Y_ALIASES
    + Z_ALIASES
    + VX_ALIASES
    + VY_ALIASES
    + VZ_ALIASES
    + SOURCE_ALIASES
)


@dataclass(frozen=True)
class SequenceClassificationResult:
    """Sequence-level UAV classification outputs."""

    predictions: pd.DataFrame
    train_features: pd.DataFrame
    predict_features: pd.DataFrame
    metrics: dict[str, Any]


def infer_sequence_class_map_from_candidates(
    candidates: CandidateFrame,
    *,
    min_confidence: float = 0.0,
    default_class: str = "unknown",
) -> dict[str, str]:
    """Infer one UAV type per sequence from weighted candidate class votes."""

    rows = candidates.rows.copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return {}
    sequence_ids = sorted(str(sequence_id) for sequence_id in rows["sequence_id"].dropna().unique())
    if "class_name" not in rows.columns:
        return {sequence_id: str(default_class) for sequence_id in sequence_ids}
    rows["confidence"] = pd.to_numeric(rows.get("confidence", 1.0), errors="coerce").fillna(1.0)
    rows = rows.loc[rows["confidence"] >= float(min_confidence)].copy()
    result: dict[str, str] = {sequence_id: str(default_class) for sequence_id in sequence_ids}
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        votes: dict[str, float] = {}
        for _, row in group.iterrows():
            label = str(row.get("class_name", default_class)).strip()
            if label.lower() in UNKNOWN_LABELS:
                continue
            votes[label] = votes.get(label, 0.0) + float(row.get("confidence", 1.0))
        if votes:
            result[str(sequence_id)] = sorted(
                votes.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
    return result


def class_map_to_frame(class_map: dict[str, str]) -> pd.DataFrame:
    """Return a stable two-column class-map table."""

    return pd.DataFrame(
        {"sequence_id": list(class_map.keys()), "uav_type": list(class_map.values())}
    ).sort_values("sequence_id").reset_index(drop=True)


def write_sequence_class_map(class_map: dict[str, str], path: Path) -> Path:
    """Write a sequence class-map CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    class_map_to_frame(class_map).to_csv(path, index=False)
    return path


def load_sequence_class_labels(path: Path) -> dict[str, str]:
    """Load one class label per sequence from class-map or official Track 5 truth."""

    path = Path(path)
    try:
        rows = load_official_track5_results_frame(path)
    except Exception:
        return load_sequence_class_map(path)
    if {"Sequence", "Classification"}.issubset(rows.columns):
        return _sequence_majority_labels(
            rows.rename(columns={"Sequence": "sequence_id", "Classification": "uav_type"})
        )
    return load_sequence_class_map(path)


def sequence_features_from_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Extract simple geometry/trajectory features for one label per sequence."""

    work = _normalize_feature_rows(rows)
    if work.empty:
        return pd.DataFrame(columns=["sequence_id"])
    records: list[dict[str, Any]] = []
    for sequence_id, group in work.groupby("sequence_id", sort=True):
        record: dict[str, Any] = {"sequence_id": str(sequence_id)}
        record["row_count"] = int(len(group))
        _add_time_features(record, group)
        _add_source_features(record, group)
        _add_numeric_stats(record, "x_m", group.get("x_m"))
        _add_numeric_stats(record, "y_m", group.get("y_m"))
        _add_numeric_stats(record, "z_m", group.get("z_m"))
        _add_numeric_stats(record, "cluster_point_count", group.get("cluster_point_count"))
        _add_numeric_stats(record, "cluster_extent_x_m", group.get("cluster_extent_x_m"))
        _add_numeric_stats(record, "cluster_extent_y_m", group.get("cluster_extent_y_m"))
        _add_numeric_stats(record, "cluster_extent_z_m", group.get("cluster_extent_z_m"))
        _add_numeric_stats(record, "cluster_extent_xy_m", group.get("cluster_extent_xy_m"))
        _add_numeric_stats(record, "cluster_extent_3d_m", group.get("cluster_extent_3d_m"))
        _add_numeric_stats(record, "cluster_density_points_per_m3", group.get("cluster_density_points_per_m3"))
        _add_numeric_stats(record, "confidence", group.get("confidence"))
        _add_numeric_stats(record, "std_xy_m", group.get("std_xy_m"))
        _add_numeric_stats(record, "std_z_m", group.get("std_z_m"))
        _add_sensor_numeric_features(record, group)
        _add_empty_radar_features(record, group)
        _add_position_features(record, group)
        _add_velocity_features(record, group)
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values("sequence_id").reset_index(drop=True)


def sequence_features_from_files(paths: list[Path]) -> pd.DataFrame:
    """Read CSV-like feature sources and extract sequence-level features."""

    raw_frames = [_read_feature_frame(path) for path in paths]
    raw_frames = [frame for frame in raw_frames if not frame.empty]
    if not raw_frames:
        return pd.DataFrame(columns=["sequence_id"])
    row_frames: list[pd.DataFrame] = []
    sequence_feature_frames: list[pd.DataFrame] = []
    for frame in raw_frames:
        if _looks_like_precomputed_sequence_features(frame):
            sequence_feature_frames.append(_normalize_precomputed_sequence_features(frame))
        else:
            row_frames.append(frame)
    if row_frames:
        sequence_feature_frames.append(
            sequence_features_from_rows(pd.concat(row_frames, ignore_index=True))
        )
    return _merge_sequence_feature_tables(sequence_feature_frames)


def classify_sequences_from_features(
    *,
    train_features: pd.DataFrame,
    train_labels: dict[str, str],
    predict_features: pd.DataFrame,
    method: str = "nearest-neighbor",
    k: int = 1,
    eval_labels: dict[str, str] | None = None,
) -> SequenceClassificationResult:
    """Fit a sequence-level baseline and predict one class per sequence."""

    method = method.strip().lower()
    train = _attach_labels(train_features, train_labels)
    if train.empty:
        raise ValueError("no training feature rows have labels")
    if predict_features.empty:
        raise ValueError("no prediction feature rows were provided")
    feature_columns = _feature_columns(train, predict_features)
    train_matrix, predict_matrix = _standardized_feature_matrices(
        train,
        predict_features,
        feature_columns,
    )
    if method == "majority":
        predictions = _predict_majority(train, predict_features)
    elif method == "nearest-neighbor":
        predictions = _predict_nearest_neighbor(
            train,
            predict_features,
            train_matrix,
            predict_matrix,
            k=max(1, int(k)),
        )
    elif method == "nearest-centroid":
        predictions = _predict_nearest_centroid(train, predict_features, train_matrix, predict_matrix)
    elif method == "logistic-regression":
        predictions = _predict_logistic_regression(train, predict_features, train_matrix, predict_matrix)
    elif method == "random-forest":
        predictions = _predict_random_forest(train, predict_features, train_matrix, predict_matrix)
    elif method in {"hist-gradient-boosting", "hist-gradient"}:
        predictions = _predict_hist_gradient_boosting(
            train, predict_features, train_matrix, predict_matrix
        )
    else:
        raise ValueError(
            "unsupported MMUAD sequence classifier method "
            f"{method!r}; expected majority, nearest-neighbor, nearest-centroid, "
            "logistic-regression, random-forest, or hist-gradient-boosting"
        )
    metrics = sequence_classification_metrics(predictions, eval_labels=eval_labels)
    metrics.update(
        {
            "method": method,
            "k": int(k),
            "train_sequence_count": int(len(train)),
            "predict_sequence_count": int(len(predict_features)),
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
        }
    )
    return SequenceClassificationResult(
        predictions=predictions,
        train_features=train,
        predict_features=predict_features,
        metrics=metrics,
    )


def write_sequence_classification_result(
    result: SequenceClassificationResult,
    *,
    output_class_map: Path,
    predictions_csv: Path | None = None,
    train_features_csv: Path | None = None,
    predict_features_csv: Path | None = None,
    metrics_json: Path | None = None,
) -> dict[str, str]:
    """Write sequence classifier artifacts."""

    paths: dict[str, str] = {}
    class_map = dict(
        zip(
            result.predictions["sequence_id"].astype(str),
            result.predictions["predicted_class"].astype(str),
            strict=False,
        )
    )
    paths["class_map_csv"] = str(write_sequence_class_map(class_map, output_class_map))
    if predictions_csv is not None:
        predictions_csv = Path(predictions_csv)
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(predictions_csv, index=False)
        paths["predictions_csv"] = str(predictions_csv)
    if train_features_csv is not None:
        train_features_csv = Path(train_features_csv)
        train_features_csv.parent.mkdir(parents=True, exist_ok=True)
        result.train_features.to_csv(train_features_csv, index=False)
        paths["train_features_csv"] = str(train_features_csv)
    if predict_features_csv is not None:
        predict_features_csv = Path(predict_features_csv)
        predict_features_csv.parent.mkdir(parents=True, exist_ok=True)
        result.predict_features.to_csv(predict_features_csv, index=False)
        paths["predict_features_csv"] = str(predict_features_csv)
    if metrics_json is not None:
        metrics_json = Path(metrics_json)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics_json.write_text(json.dumps(_jsonable(result.metrics), indent=2), encoding="utf-8")
        paths["metrics_json"] = str(metrics_json)
    return paths


def sequence_classification_metrics(
    predictions: pd.DataFrame,
    *,
    eval_labels: dict[str, str] | None,
) -> dict[str, Any]:
    """Return simple sequence-level classification metrics."""

    out: dict[str, Any] = {
        "sequence_count": int(len(predictions)),
        "prediction_counts": predictions["predicted_class"].astype(str).value_counts().to_dict(),
    }
    if not eval_labels:
        out["labels_available"] = False
        return out
    rows = predictions.copy()
    rows["ground_truth_class"] = rows["sequence_id"].astype(str).map(
        {str(key): str(value) for key, value in eval_labels.items()}
    )
    scored = rows.loc[rows["ground_truth_class"].notna()].copy()
    out["labels_available"] = True
    out["labeled_sequence_count"] = int(len(scored))
    if scored.empty:
        out["sequence_accuracy"] = None
        return out
    correct = scored["predicted_class"].astype(str) == scored["ground_truth_class"].astype(str)
    out["sequence_accuracy"] = float(correct.mean())
    out["correct_sequence_count"] = int(correct.sum())
    out["ground_truth_counts"] = scored["ground_truth_class"].astype(str).value_counts().to_dict()
    return out


def _normalize_feature_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=rows.index)
    out["sequence_id"] = _text_column(rows, SEQUENCE_ID_ALIASES, default="default")
    out["time_s"] = _numeric_column(rows, TIME_ALIASES)
    out["source"] = _text_column(rows, SOURCE_ALIASES, default="")
    out["track_id"] = _text_column(rows, TRACK_ID_ALIASES, default="")
    out["x_m"] = _numeric_column(rows, X_ALIASES)
    out["y_m"] = _numeric_column(rows, Y_ALIASES)
    out["z_m"] = _numeric_column(rows, Z_ALIASES)
    out["v_x_mps"] = _numeric_column(rows, VX_ALIASES)
    out["v_y_mps"] = _numeric_column(rows, VY_ALIASES)
    out["v_z_mps"] = _numeric_column(rows, VZ_ALIASES)
    out["confidence"] = _numeric_column(rows, CONFIDENCE_ALIASES)
    out["std_xy_m"] = _numeric_column(rows, STD_XY_ALIASES)
    out["std_z_m"] = _numeric_column(rows, STD_Z_ALIASES)
    out["cluster_point_count"] = _numeric_column(rows, CLUSTER_POINT_COUNT_ALIASES)
    out["cluster_extent_x_m"] = _numeric_column(rows, CLUSTER_EXTENT_X_ALIASES)
    out["cluster_extent_y_m"] = _numeric_column(rows, CLUSTER_EXTENT_Y_ALIASES)
    out["cluster_extent_z_m"] = _numeric_column(rows, CLUSTER_EXTENT_Z_ALIASES)
    out["cluster_extent_xy_m"] = _numeric_column(rows, CLUSTER_EXTENT_XY_ALIASES)
    out["cluster_extent_3d_m"] = _numeric_column(rows, CLUSTER_EXTENT_3D_ALIASES)
    out["cluster_density_points_per_m3"] = _numeric_column(rows, CLUSTER_DENSITY_ALIASES)
    out["empty_frame"] = _empty_frame_column(rows, out)
    finite_position = out[["x_m", "y_m", "z_m"]].notna().any(axis=1)
    finite_velocity = out[["v_x_mps", "v_y_mps", "v_z_mps"]].notna().any(axis=1)
    finite_signal = out[["confidence", "std_xy_m", "std_z_m"]].notna().any(axis=1)
    finite_cluster = out[
        [
            "cluster_point_count",
            "cluster_extent_x_m",
            "cluster_extent_y_m",
            "cluster_extent_z_m",
            "cluster_extent_xy_m",
            "cluster_extent_3d_m",
            "cluster_density_points_per_m3",
            "empty_frame",
        ]
    ].notna().any(axis=1)
    return out.loc[
        finite_position | finite_velocity | finite_signal | finite_cluster
    ].reset_index(drop=True)


def _read_feature_frame(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=path.suffix.lower() in {".jsonl", ".ndjson"})
    return pd.read_csv(path)


def _looks_like_precomputed_sequence_features(rows: pd.DataFrame) -> bool:
    if _find_column(rows, SEQUENCE_ID_ALIASES) is None:
        return False
    if any(str(column).lower().startswith("image_") for column in rows.columns):
        return True
    if any(_find_column(rows, (alias,)) is not None for alias in ROW_LEVEL_ALIASES):
        return False
    return any(
        column != "sequence_id" and pd.to_numeric(rows[column], errors="coerce").notna().any()
        for column in rows.columns
    )


def _normalize_precomputed_sequence_features(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    out["sequence_id"] = _text_column(rows, SEQUENCE_ID_ALIASES, default="default")
    for column in rows.columns:
        column_text = str(column)
        if column_text in CLASSIFIER_METADATA_COLUMNS:
            continue
        numeric = pd.to_numeric(rows[column], errors="coerce")
        if numeric.notna().any():
            out[_feature_key(column_text)] = numeric
    return (
        out.groupby("sequence_id", as_index=False)
        .mean(numeric_only=True)
        .sort_values("sequence_id")
        .reset_index(drop=True)
    )


def _merge_sequence_feature_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["sequence_id"])
    merged = frames[0].copy()
    merged["sequence_id"] = merged["sequence_id"].astype(str)
    for frame in frames[1:]:
        work = frame.copy()
        work["sequence_id"] = work["sequence_id"].astype(str)
        overlapping = set(merged.columns).intersection(work.columns).difference({"sequence_id"})
        if overlapping:
            work = work.rename(columns={column: f"{column}_extra" for column in overlapping})
        merged = merged.merge(work, on="sequence_id", how="outer")
    return merged.sort_values("sequence_id").reset_index(drop=True)


def _text_column(rows: pd.DataFrame, aliases: tuple[str, ...], *, default: str) -> pd.Series:
    column = _find_column(rows, aliases)
    if column is None:
        return pd.Series([default] * len(rows), index=rows.index)
    values = rows[column].fillna(default).astype(str).str.strip()
    return values.where(values.ne(""), default)


def _numeric_column(rows: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    column = _find_column(rows, aliases)
    if column is None:
        return pd.Series([np.nan] * len(rows), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _empty_frame_column(rows: pd.DataFrame, normalized: pd.DataFrame) -> pd.Series:
    raw_empty = _numeric_column(rows, EMPTY_FRAME_ALIASES)
    point_count = pd.to_numeric(normalized["cluster_point_count"], errors="coerce")
    empty_from_points = pd.Series(np.nan, index=rows.index, dtype=float)
    empty_from_points.loc[point_count.notna()] = (point_count.loc[point_count.notna()] <= 0).astype(
        float
    )
    return raw_empty.combine_first(empty_from_points)


def _find_column(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for alias in aliases:
        if alias in rows.columns:
            return alias
        column = lower.get(alias.lower())
        if column is not None:
            return column
    return None


def _add_time_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    time_values = pd.to_numeric(group["time_s"], errors="coerce").dropna().to_numpy(float)
    record["time_count"] = int(len(time_values))
    if time_values.size == 0:
        return
    duration = float(np.nanmax(time_values) - np.nanmin(time_values))
    record["duration_s"] = duration
    record["sample_rate_hz"] = float(len(time_values) / duration) if duration > 0 else np.nan


def _add_source_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    if "source" not in group.columns:
        return
    sources = group["source"].fillna("").astype(str).str.strip()
    sources = sources.loc[sources.ne("")]
    record["unique_source_count"] = int(sources.nunique())
    for source, count in sources.value_counts().items():
        key = _feature_key(f"source_count_{source}")
        record[key] = int(count)
        record[_feature_key(f"source_fraction_{source}")] = float(count / len(group))
    if "track_id" in group.columns:
        tracks = group["track_id"].fillna("").astype(str).str.strip()
        tracks = tracks.loc[tracks.ne("")]
        record["unique_track_count"] = int(tracks.nunique())


def _add_numeric_stats(record: dict[str, Any], prefix: str, values: Any | None) -> None:
    if values is None:
        return
    data = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return
    record[f"{prefix}_mean"] = float(np.mean(data))
    record[f"{prefix}_std"] = float(np.std(data))
    record[f"{prefix}_min"] = float(np.min(data))
    record[f"{prefix}_max"] = float(np.max(data))
    record[f"{prefix}_range"] = float(np.max(data) - np.min(data))
    record[f"{prefix}_p10"] = float(np.percentile(data, 10))
    record[f"{prefix}_p50"] = float(np.percentile(data, 50))
    record[f"{prefix}_p90"] = float(np.percentile(data, 90))
    record[f"{prefix}_p95"] = float(np.percentile(data, 95))


def _add_sensor_numeric_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    if "source" not in group.columns:
        return
    fields = (
        "cluster_point_count",
        "cluster_extent_x_m",
        "cluster_extent_y_m",
        "cluster_extent_z_m",
        "cluster_extent_xy_m",
        "cluster_extent_3d_m",
        "cluster_density_points_per_m3",
        "z_m",
    )
    for source, sensor_group in group.groupby(group["source"].astype(str), sort=True):
        source_key = _feature_key(source)
        if not source_key:
            continue
        if "time_s" in sensor_group.columns:
            sensor_times = pd.to_numeric(sensor_group["time_s"], errors="coerce").dropna()
            record[f"source_{source_key}_frame_count"] = int(sensor_times.nunique())
            if sensor_times.nunique() > 0:
                record[f"source_{source_key}_mean_rows_per_frame"] = float(
                    len(sensor_group) / sensor_times.nunique()
                )
        for field in fields:
            _add_numeric_stats(record, f"source_{source_key}_{field}", sensor_group.get(field))
        xyz = sensor_group[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
        valid = xyz.notna().all(axis=1)
        if valid.any():
            _add_numeric_stats(
                record,
                f"source_{source_key}_range_xy_m",
                np.hypot(xyz.loc[valid, "x_m"], xyz.loc[valid, "y_m"]),
            )
            _add_numeric_stats(
                record,
                f"source_{source_key}_range_3d_m",
                np.linalg.norm(xyz.loc[valid, ["x_m", "y_m", "z_m"]].to_numpy(float), axis=1),
            )


def _add_empty_radar_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    if "source" not in group.columns or "time_s" not in group.columns:
        return
    radar_rows = group.loc[group["source"].astype(str).str.contains("radar", case=False, na=False)]
    if radar_rows.empty:
        return
    empty = pd.to_numeric(radar_rows.get("empty_frame"), errors="coerce")
    if empty.notna().any():
        by_time = (
            radar_rows.assign(_empty_frame=empty)
            .groupby("time_s", sort=True)["_empty_frame"]
            .max()
        )
        record["radar_frame_count"] = int(len(by_time))
        record["radar_empty_frame_count"] = int((by_time >= 0.5).sum())
        record["radar_empty_frame_fraction"] = float((by_time >= 0.5).mean())


def _add_position_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    xyz = group[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    valid = xyz.notna().all(axis=1)
    if not valid.any():
        return
    xyz = xyz.loc[valid].copy()
    _add_numeric_stats(record, "range_xy_m", np.hypot(xyz["x_m"], xyz["y_m"]))
    _add_numeric_stats(
        record,
        "range_3d_m",
        np.linalg.norm(xyz[["x_m", "y_m", "z_m"]].to_numpy(float), axis=1),
    )
    times = pd.to_numeric(group.loc[valid, "time_s"], errors="coerce")
    order = np.argsort(times.to_numpy(float)) if times.notna().all() else np.arange(len(xyz))
    ordered_xyz = xyz.to_numpy(float)[order]
    if ordered_xyz.shape[0] >= 2:
        displacement = ordered_xyz[-1] - ordered_xyz[0]
        record["trajectory_displacement_2d_m"] = float(np.linalg.norm(displacement[:2]))
        record["trajectory_displacement_3d_m"] = float(np.linalg.norm(displacement))
        segment_lengths = np.linalg.norm(np.diff(ordered_xyz, axis=0), axis=1)
        record["trajectory_path_length_3d_m"] = float(np.sum(segment_lengths))
        if times.notna().all():
            ordered_t = times.to_numpy(float)[order]
            dt = np.diff(ordered_t)
            valid_dt = dt > 0
            if valid_dt.any():
                diff_speed = segment_lengths[valid_dt] / dt[valid_dt]
                _add_numeric_stats(record, "diff_speed_mps", pd.Series(diff_speed))


def _add_velocity_features(record: dict[str, Any], group: pd.DataFrame) -> None:
    velocity = group[["v_x_mps", "v_y_mps", "v_z_mps"]].apply(pd.to_numeric, errors="coerce")
    valid = velocity.notna().all(axis=1)
    if not valid.any():
        return
    values = velocity.loc[valid].to_numpy(float)
    _add_numeric_stats(record, "speed_xy_mps", pd.Series(np.linalg.norm(values[:, :2], axis=1)))
    _add_numeric_stats(record, "speed_3d_mps", pd.Series(np.linalg.norm(values, axis=1)))
    _add_numeric_stats(record, "vertical_speed_mps", pd.Series(values[:, 2]))


def _attach_labels(features: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = features.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    label_map = {str(key): str(value) for key, value in labels.items()}
    rows["uav_type"] = rows["sequence_id"].map(label_map)
    return rows.loc[rows["uav_type"].notna()].reset_index(drop=True)


def _feature_columns(train: pd.DataFrame, predict: pd.DataFrame) -> list[str]:
    columns = sorted(
        set(train.columns)
        .union(predict.columns)
        .difference(CLASSIFIER_METADATA_COLUMNS)
    )
    numeric_columns: list[str] = []
    for column in columns:
        train_numeric = _numeric_feature_series(train, column)
        predict_numeric = _numeric_feature_series(predict, column)
        if train_numeric.notna().any() or predict_numeric.notna().any():
            numeric_columns.append(str(column))
    if not numeric_columns:
        raise ValueError("sequence classifier found no numeric feature columns")
    return numeric_columns


def _standardized_feature_matrices(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix = _numeric_matrix(train, feature_columns)
    predict_matrix = _numeric_matrix(predict, feature_columns)
    means = np.nanmean(train_matrix, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    train_matrix = np.where(np.isfinite(train_matrix), train_matrix, means)
    predict_matrix = np.where(np.isfinite(predict_matrix), predict_matrix, means)
    scale = np.nanstd(train_matrix, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-9), scale, 1.0)
    return (train_matrix - means) / scale, (predict_matrix - means) / scale


def _numeric_matrix(rows: pd.DataFrame, columns: list[str]) -> np.ndarray:
    numeric = {
        column: _numeric_feature_series(rows, column)
        for column in columns
    }
    return pd.DataFrame(numeric, index=rows.index).to_numpy(float)


def _numeric_feature_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series([np.nan] * len(rows), index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _predict_majority(train: pd.DataFrame, predict_features: pd.DataFrame) -> pd.DataFrame:
    label = _majority_label(train["uav_type"])
    return pd.DataFrame(
        {
            "sequence_id": predict_features["sequence_id"].astype(str),
            "predicted_class": label,
            "class_source": "sequence-majority",
            "nearest_train_sequence_id": "",
            "nearest_distance": np.nan,
        }
    )


def _predict_nearest_neighbor(
    train: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_matrix: np.ndarray,
    predict_matrix: np.ndarray,
    *,
    k: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    labels = train["uav_type"].astype(str).tolist()
    sequence_ids = train["sequence_id"].astype(str).tolist()
    for row_idx, sequence_id in enumerate(predict_features["sequence_id"].astype(str)):
        distances = np.linalg.norm(train_matrix - predict_matrix[row_idx], axis=1)
        nearest = np.argsort(distances)[: min(k, len(distances))]
        predicted = _weighted_majority_label([labels[idx] for idx in nearest], distances[nearest])
        best = int(nearest[0])
        records.append(
            {
                "sequence_id": sequence_id,
                "predicted_class": predicted,
                "class_source": f"sequence-nearest-neighbor-k{k}",
                "nearest_train_sequence_id": sequence_ids[best],
                "nearest_distance": float(distances[best]),
            }
        )
    return pd.DataFrame.from_records(records)


def _predict_nearest_centroid(
    train: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_matrix: np.ndarray,
    predict_matrix: np.ndarray,
) -> pd.DataFrame:
    labels = train["uav_type"].astype(str).to_numpy()
    centroids = []
    for label in sorted(set(labels)):
        centroids.append((label, train_matrix[labels == label].mean(axis=0)))
    records: list[dict[str, Any]] = []
    for row_idx, sequence_id in enumerate(predict_features["sequence_id"].astype(str)):
        distances = [
            (label, float(np.linalg.norm(predict_matrix[row_idx] - centroid)))
            for label, centroid in centroids
        ]
        predicted, distance = sorted(distances, key=lambda item: (item[1], item[0]))[0]
        records.append(
            {
                "sequence_id": sequence_id,
                "predicted_class": predicted,
                "class_source": "sequence-nearest-centroid",
                "nearest_train_sequence_id": "",
                "nearest_distance": distance,
            }
        )
    return pd.DataFrame.from_records(records)


def _predict_logistic_regression(
    train: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_matrix: np.ndarray,
    predict_matrix: np.ndarray,
) -> pd.DataFrame:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - depends on optional sklearn
        raise ValueError("logistic-regression sequence classifier requires scikit-learn") from exc
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_matrix, train["uav_type"].astype(str))
    return _predict_sklearn_model(
        model,
        predict_features,
        predict_matrix,
        class_source="sequence-logistic-regression",
    )


def _predict_random_forest(
    train: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_matrix: np.ndarray,
    predict_matrix: np.ndarray,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:  # pragma: no cover - depends on optional sklearn
        raise ValueError("random-forest sequence classifier requires scikit-learn") from exc
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=0,
        class_weight="balanced",
        min_samples_leaf=1,
    )
    model.fit(train_matrix, train["uav_type"].astype(str))
    return _predict_sklearn_model(
        model,
        predict_features,
        predict_matrix,
        class_source="sequence-random-forest",
    )


def _predict_hist_gradient_boosting(
    train: pd.DataFrame,
    predict_features: pd.DataFrame,
    train_matrix: np.ndarray,
    predict_matrix: np.ndarray,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except Exception as exc:  # pragma: no cover - depends on optional sklearn
        raise ValueError(
            "hist-gradient-boosting sequence classifier requires scikit-learn"
        ) from exc
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        min_samples_leaf=1,
        random_state=0,
    )
    model.fit(train_matrix, train["uav_type"].astype(str))
    return _predict_sklearn_model(
        model,
        predict_features,
        predict_matrix,
        class_source="sequence-hist-gradient-boosting",
    )


def _predict_sklearn_model(
    model: Any,
    predict_features: pd.DataFrame,
    predict_matrix: np.ndarray,
    *,
    class_source: str,
) -> pd.DataFrame:
    predicted = model.predict(predict_matrix)
    confidence = (
        np.max(model.predict_proba(predict_matrix), axis=1)
        if hasattr(model, "predict_proba")
        else [np.nan] * len(predicted)
    )
    return pd.DataFrame(
        {
            "sequence_id": predict_features["sequence_id"].astype(str),
            "predicted_class": [str(label) for label in predicted],
            "class_source": class_source,
            "nearest_train_sequence_id": "",
            "nearest_distance": np.nan,
            "classification_confidence": confidence,
        }
    )


def _majority_label(labels: pd.Series) -> str:
    counts = labels.astype(str).value_counts()
    return sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0][0]


def _weighted_majority_label(labels: list[str], distances: np.ndarray) -> str:
    weights: dict[str, float] = {}
    for label, distance in zip(labels, distances, strict=False):
        weights[str(label)] = weights.get(str(label), 0.0) + 1.0 / max(float(distance), 1.0e-9)
    return sorted(weights.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _sequence_majority_labels(rows: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    for sequence_id, group in rows.groupby(rows["sequence_id"].astype(str), sort=True):
        labels[str(sequence_id)] = _majority_label(group["uav_type"].astype(str))
    return labels


def _feature_key(text: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(text)).strip("_")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
