"""Sequence-level MMUAD point-cloud alignment and extraction diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.coordinate_alignment_audit import (
    AlignmentVariant,
    _base_variants,
    _finite_candidate_rows,
    _finite_truth_rows,
    _fraction,
    _mean_or_nan,
    _median_translation,
    _nearest_truth_distances,
    _parse_max_time_delta,
    _percentile_or_nan,
    _sensor_from_path,
    _threshold_fraction,
    _transform_candidates,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_point_cloud_file_as_candidates, merge_candidate_frames
from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.sequence import _point_cloud_frame_time_s, discover_sequence_paths


@dataclass(frozen=True)
class SensorExtractionSummary:
    """Point-cloud extraction stats for one sequence/sensor."""

    candidates: CandidateFrame
    source_frame_count: int
    loaded_source_frame_count: int
    load_error_frame_count: int
    empty_frame_count: int
    source_frame_times_s: tuple[float, ...]
    candidates_per_source_frame: tuple[int, ...]


def build_sequence_alignment_audit(
    sequence_root: Path,
    truth_path: Path,
    *,
    sequence_glob: str = "*",
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
    max_time_delta_s: float | None = 0.5,
    include_translation_diagnostic: bool = True,
    scales: tuple[float, ...] = (0.001, 0.01, 1.0),
) -> pd.DataFrame:
    """Audit whether weak MMUAD sequences fail at timing, frame, or extraction level."""

    truth = load_evaluation_truth_file(Path(truth_path)).rows
    truth = _finite_truth_rows(truth)
    sequences = discover_sequence_paths(Path(sequence_root), sequence_glob=sequence_glob)
    records: list[dict[str, Any]] = []
    base_variants = _base_variants(scales=scales)
    for paths in sequences:
        truth_sequence = truth.loc[truth["sequence_id"].astype(str) == paths.sequence_id].copy()
        if truth_sequence.empty:
            continue
        sensor_extractions = _load_sequence_sensor_extractions(
            paths,
            voxel_size_m=voxel_size_m,
            min_cluster_points=min_cluster_points,
        )
        for sensor, extraction in sorted(sensor_extractions.items()):
            candidate_rows = _finite_candidate_rows(extraction.candidates.rows)
            extraction_stats = _extraction_stats(
                truth_sequence,
                candidate_rows,
                extraction,
                voxel_size_m=voxel_size_m,
                min_cluster_points=min_cluster_points,
                max_time_delta_s=max_time_delta_s,
            )
            for variant in base_variants:
                records.append(
                    _audit_alignment_variant(
                        truth_sequence,
                        candidate_rows,
                        sequence_id=paths.sequence_id,
                        sensor=sensor,
                        variant=variant,
                        max_time_delta_s=max_time_delta_s,
                        extraction_stats=extraction_stats,
                    )
                )
                if include_translation_diagnostic:
                    translated = AlignmentVariant(
                        name=f"{variant.name}+median-translation",
                        permutation=variant.permutation,
                        signs=variant.signs,
                        scale=variant.scale,
                        translation_mode="per-sequence-median-diagnostic",
                    )
                    records.append(
                        _audit_alignment_variant(
                            truth_sequence,
                            candidate_rows,
                            sequence_id=paths.sequence_id,
                            sensor=sensor,
                            variant=translated,
                            max_time_delta_s=max_time_delta_s,
                            extraction_stats=extraction_stats,
                        )
                    )
    return pd.DataFrame.from_records(records, columns=_audit_columns())


def write_sequence_alignment_audit(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``mmuad_sequence_alignment_audit.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_sequence_alignment_decision_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize sequence/sensor alignment evidence into parser decisions."""

    rows = pd.DataFrame(audit).copy()
    if rows.empty:
        return pd.DataFrame(columns=_decision_summary_columns())
    records: list[dict[str, Any]] = []
    for (sequence_id, sensor), group in rows.groupby(["sequence_id", "sensor"], sort=True):
        as_is = _pick_variant_row(group, variant="as-is")
        translated = _pick_best_translation_row(group)
        if as_is is None:
            as_is = group.iloc[0]
        if translated is None:
            translated = as_is
        truth_frame_count = _numeric_value(as_is, "truth_frame_count")
        within_5 = _frame_count_from_fraction(
            _numeric_value(as_is, "fraction_frames_with_cluster_within_5m"),
            truth_frame_count,
        )
        within_10 = _frame_count_from_fraction(
            _numeric_value(as_is, "fraction_frames_with_cluster_within_10m"),
            truth_frame_count,
        )
        within_20 = _frame_count_from_fraction(
            _numeric_value(as_is, "fraction_frames_with_cluster_within_20m"),
            truth_frame_count,
        )
        translation = np.asarray(
            [
                _numeric_value(translated, "translation_x_m"),
                _numeric_value(translated, "translation_y_m"),
                _numeric_value(translated, "translation_z_m"),
            ],
            dtype=float,
        )
        if not np.isfinite(translation).all():
            translation = np.zeros(3, dtype=float)
        record = {
            "sequence": str(sequence_id),
            "sensor": str(sensor),
            "as_is_nearest_mean": _numeric_value(as_is, "mean_nearest_cluster_to_truth_distance_m"),
            "as_is_nearest_p95": _numeric_value(as_is, "p95_nearest_cluster_to_truth_distance_m"),
            "median_translation_vector_x": float(translation[0]),
            "median_translation_vector_y": float(translation[1]),
            "median_translation_vector_z": float(translation[2]),
            "median_translation_norm": float(np.linalg.norm(translation)),
            "after_translation_nearest_mean": _numeric_value(
                translated,
                "mean_nearest_cluster_to_truth_distance_m",
            ),
            "after_translation_nearest_p95": _numeric_value(
                translated,
                "p95_nearest_cluster_to_truth_distance_m",
            ),
            "frames_with_candidate_within_5m": within_5,
            "frames_with_candidate_within_10m": within_10,
            "frames_with_candidate_within_20m": within_20,
            "radar_raw_point_count": _radar_value(as_is, "raw_point_count"),
            "radar_candidate_count": _radar_value(as_is, "candidate_count"),
        }
        record["diagnosis"] = _diagnose_alignment(group, as_is, translated)
        records.append(record)
    return pd.DataFrame.from_records(records, columns=_decision_summary_columns())


def write_sequence_alignment_decision_summary(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``mmuad_sequence_alignment_decision_summary.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _load_sequence_sensor_extractions(
    paths,
    *,
    voxel_size_m: float,
    min_cluster_points: int,
) -> dict[str, SensorExtractionSummary]:
    frames_by_sensor: dict[str, list[CandidateFrame]] = {}
    source_frame_count: dict[str, int] = {}
    loaded_source_frame_count: dict[str, int] = {}
    load_error_frame_count: dict[str, int] = {}
    empty_frame_count: dict[str, int] = {}
    source_frame_times: dict[str, list[float]] = {}
    candidates_per_frame: dict[str, list[int]] = {}
    for path in paths.point_cloud_files:
        sensor = _sensor_from_path(path, sequence_root=paths.root)
        source_frame_count[sensor] = source_frame_count.get(sensor, 0) + 1
        time_s = _point_cloud_frame_time_s(path, sequence_root=paths.root)
        if time_s is not None and np.isfinite(float(time_s)):
            source_frame_times.setdefault(sensor, []).append(float(time_s))
        try:
            frame = load_point_cloud_file_as_candidates(
                path,
                source=sensor,
                sequence_id=paths.sequence_id,
                time_s=time_s,
                voxel_size_m=voxel_size_m,
                min_points=min_cluster_points,
            )
        except Exception:
            load_error_frame_count[sensor] = load_error_frame_count.get(sensor, 0) + 1
            continue
        loaded_source_frame_count[sensor] = loaded_source_frame_count.get(sensor, 0) + 1
        rows = _finite_candidate_rows(frame.rows)
        if not rows.empty:
            frames_by_sensor.setdefault(sensor, []).append(CandidateFrame(rows))
        candidate_count = int(len(rows))
        candidates_per_frame.setdefault(sensor, []).append(candidate_count)
        if candidate_count <= 0:
            empty_frame_count[sensor] = empty_frame_count.get(sensor, 0) + 1

    sensors = sorted(source_frame_count)
    summaries: dict[str, SensorExtractionSummary] = {}
    for sensor in sensors:
        frames = frames_by_sensor.get(sensor, [])
        candidates = merge_candidate_frames(frames)
        summaries[sensor] = SensorExtractionSummary(
            candidates=candidates,
            source_frame_count=source_frame_count.get(sensor, 0),
            loaded_source_frame_count=loaded_source_frame_count.get(sensor, 0),
            load_error_frame_count=load_error_frame_count.get(sensor, 0),
            empty_frame_count=empty_frame_count.get(sensor, 0),
            source_frame_times_s=tuple(source_frame_times.get(sensor, ())),
            candidates_per_source_frame=tuple(candidates_per_frame.get(sensor, ())),
        )
    return summaries


def _extraction_stats(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    extraction: SensorExtractionSummary,
    *,
    voxel_size_m: float,
    min_cluster_points: int,
    max_time_delta_s: float | None,
) -> dict[str, Any]:
    source_time_deltas = _nearest_source_frame_time_deltas(
        truth,
        extraction.source_frame_times_s,
    )
    finite_source_time = source_time_deltas[np.isfinite(source_time_deltas)]
    if max_time_delta_s is None:
        source_time_matched = np.isfinite(source_time_deltas)
    else:
        source_time_matched = np.isfinite(source_time_deltas) & (
            source_time_deltas <= float(max_time_delta_s)
        )
    candidate_times = _numeric_values(candidates, "time_s")
    point_counts = _numeric_values(candidates, "cluster_point_count")
    ranges = _candidate_range_values(candidates)
    heights = _numeric_values(candidates, "cluster_height_m")
    if heights.size == 0:
        heights = _numeric_values(candidates, "z_m")
    extents = _numeric_values(candidates, "cluster_extent_3d_m")
    per_frame = np.asarray(extraction.candidates_per_source_frame, dtype=float)
    per_frame = per_frame[np.isfinite(per_frame)]
    return {
        "voxel_size_m": float(voxel_size_m),
        "min_cluster_points": int(min_cluster_points),
        "max_time_delta_s": float(max_time_delta_s) if max_time_delta_s is not None else np.nan,
        "truth_frame_count": int(len(truth)),
        "source_frame_count": int(extraction.source_frame_count),
        "loaded_source_frame_count": int(extraction.loaded_source_frame_count),
        "load_error_frame_count": int(extraction.load_error_frame_count),
        "empty_frame_count": int(extraction.empty_frame_count),
        "no_candidate_source_frame_count": int(
            extraction.empty_frame_count + extraction.load_error_frame_count
        ),
        "candidate_frame_count": int(len(np.unique(candidate_times))) if candidate_times.size else 0,
        "candidate_count": int(len(candidates)),
        "raw_point_count": int(np.nansum(point_counts)) if point_counts.size else 0,
        "source_frame_with_candidates_fraction": _fraction(
            extraction.source_frame_count
            - extraction.empty_frame_count
            - extraction.load_error_frame_count,
            extraction.source_frame_count,
        ),
        "mean_nearest_source_frame_abs_time_delta_s": _mean_or_nan(finite_source_time),
        "p95_nearest_source_frame_abs_time_delta_s": _percentile_or_nan(
            finite_source_time,
            95.0,
        ),
        "source_time_matched_truth_frame_count": int(source_time_matched.sum()),
        "source_time_matched_truth_frame_fraction": _fraction(
            source_time_matched.sum(),
            len(truth),
        ),
        "candidates_per_source_frame_mean": _mean_or_nan(per_frame),
        "candidates_per_source_frame_p95": _percentile_or_nan(per_frame, 95.0),
        "candidates_per_source_frame_max": float(np.max(per_frame)) if per_frame.size else np.nan,
        "cluster_point_count_min": float(np.min(point_counts)) if point_counts.size else np.nan,
        "cluster_point_count_mean": _mean_or_nan(point_counts),
        "cluster_point_count_median": _percentile_or_nan(point_counts, 50.0),
        "cluster_point_count_p95": _percentile_or_nan(point_counts, 95.0),
        "cluster_point_count_max": float(np.max(point_counts)) if point_counts.size else np.nan,
        "cluster_range_3d_m_mean": _mean_or_nan(ranges),
        "cluster_range_3d_m_p95": _percentile_or_nan(ranges, 95.0),
        "cluster_height_m_mean": _mean_or_nan(heights),
        "cluster_height_m_p95": _percentile_or_nan(heights, 95.0),
        "cluster_extent_3d_m_mean": _mean_or_nan(extents),
        "cluster_extent_3d_m_p95": _percentile_or_nan(extents, 95.0),
    }


def _audit_alignment_variant(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    sequence_id: str,
    sensor: str,
    variant: AlignmentVariant,
    max_time_delta_s: float | None,
    extraction_stats: dict[str, Any],
) -> dict[str, Any]:
    transformed = _transform_candidates(candidates, variant)
    translation = np.zeros(3, dtype=float)
    if variant.translation_mode == "per-sequence-median-diagnostic":
        translation = _median_translation(truth, transformed, max_time_delta_s=max_time_delta_s)
        transformed = transformed.copy()
        transformed[["audit_x_m", "audit_y_m", "audit_z_m"]] = (
            transformed[["audit_x_m", "audit_y_m", "audit_z_m"]].to_numpy(float) + translation
        )
    distances, time_deltas, matched = _nearest_truth_distances(
        truth,
        transformed,
        max_time_delta_s=max_time_delta_s,
    )
    finite_distances = distances[np.isfinite(distances)]
    finite_time = np.abs(time_deltas[np.isfinite(time_deltas)])
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "sensor": sensor,
        "variant": variant.name,
        "axis_permutation": variant.axis_permutation,
        "axis_sign": variant.axis_sign,
        "scale": float(variant.scale),
        "translation_mode": variant.translation_mode,
        "translation_x_m": float(translation[0]),
        "translation_y_m": float(translation[1]),
        "translation_z_m": float(translation[2]),
        **extraction_stats,
        "matched_truth_frame_count": int(matched.sum()),
        "matched_truth_frame_fraction": _fraction(matched.sum(), len(truth)),
        "mean_abs_time_delta_s": _mean_or_nan(finite_time),
        "p95_abs_time_delta_s": _percentile_or_nan(finite_time, 95.0),
        "mean_nearest_cluster_to_truth_distance_m": _mean_or_nan(finite_distances),
        "p95_nearest_cluster_to_truth_distance_m": _percentile_or_nan(finite_distances, 95.0),
        "fraction_frames_with_cluster_within_5m": _threshold_fraction(distances, 5.0, len(truth)),
        "fraction_frames_with_cluster_within_10m": _threshold_fraction(distances, 10.0, len(truth)),
        "fraction_frames_with_cluster_within_20m": _threshold_fraction(distances, 20.0, len(truth)),
    }
    return record


def _nearest_source_frame_time_deltas(
    truth: pd.DataFrame,
    source_frame_times_s: tuple[float, ...],
) -> np.ndarray:
    deltas = np.full(len(truth), np.nan, dtype=float)
    if not source_frame_times_s:
        return deltas
    source_times = np.asarray(source_frame_times_s, dtype=float)
    source_times = source_times[np.isfinite(source_times)]
    if source_times.size == 0:
        return deltas
    for idx, (_, truth_row) in enumerate(truth.sort_values("time_s").iterrows()):
        truth_time = float(truth_row["time_s"])
        deltas[idx] = float(np.min(np.abs(source_times - truth_time)))
    return deltas


def _numeric_values(rows: pd.DataFrame, column: str) -> np.ndarray:
    if rows.empty or column not in rows.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def _candidate_range_values(rows: pd.DataFrame) -> np.ndarray:
    ranges = _numeric_values(rows, "cluster_range_3d_m")
    if ranges.size or rows.empty:
        return ranges
    xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    return np.linalg.norm(xyz[finite], axis=1)


def _audit_columns() -> list[str]:
    return [
        "sequence_id",
        "sensor",
        "variant",
        "axis_permutation",
        "axis_sign",
        "scale",
        "translation_mode",
        "translation_x_m",
        "translation_y_m",
        "translation_z_m",
        "voxel_size_m",
        "min_cluster_points",
        "max_time_delta_s",
        "truth_frame_count",
        "source_frame_count",
        "loaded_source_frame_count",
        "load_error_frame_count",
        "empty_frame_count",
        "no_candidate_source_frame_count",
        "candidate_frame_count",
        "candidate_count",
        "raw_point_count",
        "source_frame_with_candidates_fraction",
        "mean_nearest_source_frame_abs_time_delta_s",
        "p95_nearest_source_frame_abs_time_delta_s",
        "source_time_matched_truth_frame_count",
        "source_time_matched_truth_frame_fraction",
        "candidates_per_source_frame_mean",
        "candidates_per_source_frame_p95",
        "candidates_per_source_frame_max",
        "cluster_point_count_min",
        "cluster_point_count_mean",
        "cluster_point_count_median",
        "cluster_point_count_p95",
        "cluster_point_count_max",
        "cluster_range_3d_m_mean",
        "cluster_range_3d_m_p95",
        "cluster_height_m_mean",
        "cluster_height_m_p95",
        "cluster_extent_3d_m_mean",
        "cluster_extent_3d_m_p95",
        "matched_truth_frame_count",
        "matched_truth_frame_fraction",
        "mean_abs_time_delta_s",
        "p95_abs_time_delta_s",
        "mean_nearest_cluster_to_truth_distance_m",
        "p95_nearest_cluster_to_truth_distance_m",
        "fraction_frames_with_cluster_within_5m",
        "fraction_frames_with_cluster_within_10m",
        "fraction_frames_with_cluster_within_20m",
    ]


def _decision_summary_columns() -> list[str]:
    return [
        "sequence",
        "sensor",
        "as_is_nearest_mean",
        "as_is_nearest_p95",
        "median_translation_vector_x",
        "median_translation_vector_y",
        "median_translation_vector_z",
        "median_translation_norm",
        "after_translation_nearest_mean",
        "after_translation_nearest_p95",
        "frames_with_candidate_within_5m",
        "frames_with_candidate_within_10m",
        "frames_with_candidate_within_20m",
        "radar_raw_point_count",
        "radar_candidate_count",
        "diagnosis",
    ]


def _pick_variant_row(group: pd.DataFrame, *, variant: str) -> pd.Series | None:
    rows = group.loc[group["variant"].astype(str) == variant]
    if rows.empty:
        return None
    return rows.iloc[0]


def _pick_best_translation_row(group: pd.DataFrame) -> pd.Series | None:
    translated = group.loc[
        group["translation_mode"].astype(str) == "per-sequence-median-diagnostic"
    ].copy()
    if translated.empty:
        return _pick_variant_row(group, variant="as-is+median-translation")
    translated["_sort_p95"] = pd.to_numeric(
        translated.get("p95_nearest_cluster_to_truth_distance_m"),
        errors="coerce",
    )
    translated["_sort_mean"] = pd.to_numeric(
        translated.get("mean_nearest_cluster_to_truth_distance_m"),
        errors="coerce",
    )
    translated = translated.sort_values(
        ["_sort_p95", "_sort_mean"],
        na_position="last",
    )
    return translated.iloc[0]


def _numeric_value(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def _frame_count_from_fraction(fraction: float, truth_frame_count: float) -> int:
    if not np.isfinite(fraction) or not np.isfinite(truth_frame_count):
        return 0
    return int(round(float(fraction) * float(truth_frame_count)))


def _radar_value(row: pd.Series, column: str) -> float:
    sensor = str(row.get("sensor", "")).lower()
    if "radar" not in sensor:
        return float("nan")
    return _numeric_value(row, column)


def _diagnose_alignment(
    group: pd.DataFrame,
    as_is: pd.Series,
    translated: pd.Series,
) -> str:
    candidate_count = _numeric_value(as_is, "candidate_count")
    source_frame_count = _numeric_value(as_is, "source_frame_count")
    source_match_fraction = _numeric_value(as_is, "source_time_matched_truth_frame_fraction")
    source_candidate_fraction = _numeric_value(as_is, "source_frame_with_candidates_fraction")
    as_is_mean = _numeric_value(as_is, "mean_nearest_cluster_to_truth_distance_m")
    as_is_p95 = _numeric_value(as_is, "p95_nearest_cluster_to_truth_distance_m")
    translated_p95 = _numeric_value(translated, "p95_nearest_cluster_to_truth_distance_m")
    translated_mean = _numeric_value(translated, "mean_nearest_cluster_to_truth_distance_m")
    within_20 = _numeric_value(as_is, "fraction_frames_with_cluster_within_20m")
    translation_norm = float(
        np.linalg.norm(
            [
                _numeric_value(translated, "translation_x_m"),
                _numeric_value(translated, "translation_y_m"),
                _numeric_value(translated, "translation_z_m"),
            ]
        )
    )
    if not np.isfinite(candidate_count) or candidate_count <= 0:
        return "candidate_extraction_empty"
    if (
        not np.isfinite(source_frame_count)
        or source_frame_count <= 0
        or (np.isfinite(source_match_fraction) and source_match_fraction < 0.5)
        or (np.isfinite(source_candidate_fraction) and source_candidate_fraction < 0.5)
    ):
        return "missing_sensor_frames"
    if _is_good_alignment(as_is_mean, as_is_p95):
        return "as_is_good"
    if (
        _is_good_alignment(translated_mean, translated_p95)
        and np.isfinite(translation_norm)
        and translation_norm >= 5.0
    ):
        return "translation_offset_suspected"
    best_nontranslation_p95 = _best_nontranslation_p95(group)
    if (
        np.isfinite(best_nontranslation_p95)
        and np.isfinite(as_is_p95)
        and best_nontranslation_p95 < min(as_is_p95 * 0.7, as_is_p95 - 5.0)
    ):
        return "axis_or_scale_suspected"
    if np.isfinite(within_20) and within_20 >= 0.5:
        return "ranker_problem"
    return "axis_or_scale_suspected"


def _is_good_alignment(mean_m: float, p95_m: float) -> bool:
    return (
        np.isfinite(mean_m)
        and np.isfinite(p95_m)
        and (mean_m <= 5.0 or p95_m <= 10.0)
    )


def _best_nontranslation_p95(group: pd.DataFrame) -> float:
    variants = group.loc[group["translation_mode"].astype(str) != "per-sequence-median-diagnostic"]
    values = pd.to_numeric(
        variants.get("p95_nearest_cluster_to_truth_distance_m"),
        errors="coerce",
    )
    values = values.loc[np.isfinite(values)]
    return float(values.min()) if len(values) else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-sequence-alignment-audit",
        description="audit MMUAD per-sequence point-cloud extraction, timing, and alignment",
    )
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--truth-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--decision-summary-csv", type=Path)
    parser.add_argument("--sequence-glob", default="seq0002")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--max-time-delta-s", default="0.5", type=_parse_max_time_delta)
    parser.add_argument("--scale", action="append", type=float, dest="scales")
    parser.add_argument("--skip-translation-diagnostic", action="store_true")
    args = parser.parse_args(argv)

    output_csv = args.output_csv
    if output_csv is None:
        if args.output_dir is None:
            raise SystemExit("provide --output-dir or --output-csv")
        output_csv = args.output_dir / "mmuad_sequence_alignment_audit.csv"
    scales = tuple(args.scales) if args.scales else (0.001, 0.01, 1.0)
    audit = build_sequence_alignment_audit(
        args.sequence_root,
        args.truth_file,
        sequence_glob=args.sequence_glob,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
        max_time_delta_s=args.max_time_delta_s,
        include_translation_diagnostic=not args.skip_translation_diagnostic,
        scales=scales,
    )
    path = write_sequence_alignment_audit(audit, output_csv)
    decision_csv = args.decision_summary_csv
    if decision_csv is None:
        decision_csv = Path(output_csv).parent / "mmuad_sequence_alignment_decision_summary.csv"
    decision = build_sequence_alignment_decision_summary(audit)
    decision_path = write_sequence_alignment_decision_summary(decision, decision_csv)
    print("mmuad_sequence_alignment_audit=ok")
    print(f"output_csv={path}")
    print(f"decision_summary_csv={decision_path}")
    print(f"rows={len(audit)}")
    print(f"decision_rows={len(decision)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
