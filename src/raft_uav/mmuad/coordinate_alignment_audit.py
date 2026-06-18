"""Coordinate-frame and time-alignment diagnostics for MMUAD point clouds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_point_cloud_file_as_candidates, merge_candidate_frames
from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.sequence import _point_cloud_frame_time_s, discover_sequence_paths


@dataclass(frozen=True)
class AlignmentVariant:
    name: str
    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]
    scale: float
    translation_mode: str = "none"

    @property
    def axis_permutation(self) -> str:
        names = ("x", "y", "z")
        return ",".join(names[index] for index in self.permutation)

    @property
    def axis_sign(self) -> str:
        return ",".join("+" if sign >= 0 else "-" for sign in self.signs)


def build_coordinate_alignment_audit(
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
    """Compare raw point-cloud clusters against public validation truth."""

    truth = load_evaluation_truth_file(Path(truth_path)).rows
    truth = _finite_truth_rows(truth)
    sequences = discover_sequence_paths(Path(sequence_root), sequence_glob=sequence_glob)
    records: list[dict[str, Any]] = []
    base_variants = _base_variants(scales=scales)
    for paths in sequences:
        truth_sequence = truth.loc[truth["sequence_id"].astype(str) == paths.sequence_id].copy()
        if truth_sequence.empty:
            continue
        sensor_frames = _load_sequence_sensor_candidates(
            paths,
            voxel_size_m=voxel_size_m,
            min_cluster_points=min_cluster_points,
        )
        for sensor, candidates in sorted(sensor_frames.items()):
            candidate_rows = _finite_candidate_rows(candidates.rows)
            for variant in base_variants:
                records.append(
                    _audit_variant(
                        truth_sequence,
                        candidate_rows,
                        sequence_id=paths.sequence_id,
                        sensor=sensor,
                        variant=variant,
                        max_time_delta_s=max_time_delta_s,
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
                        _audit_variant(
                            truth_sequence,
                            candidate_rows,
                            sequence_id=paths.sequence_id,
                            sensor=sensor,
                            variant=translated,
                            max_time_delta_s=max_time_delta_s,
                        )
                    )
    return pd.DataFrame.from_records(records, columns=_audit_columns())


def write_coordinate_alignment_audit(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``mmuad_coordinate_alignment_audit.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _base_variants(*, scales: tuple[float, ...]) -> list[AlignmentVariant]:
    variants = [
        AlignmentVariant("as-is", (0, 1, 2), (1, 1, 1), 1.0),
        AlignmentVariant("x-y-swap", (1, 0, 2), (1, 1, 1), 1.0),
        AlignmentVariant("y-sign-flip", (0, 1, 2), (1, -1, 1), 1.0),
        AlignmentVariant("z-sign-flip", (0, 1, 2), (1, 1, -1), 1.0),
    ]
    seen = {(variant.permutation, variant.signs, float(variant.scale), variant.name) for variant in variants}
    for scale in scales:
        variant = AlignmentVariant(f"scale-{float(scale):g}", (0, 1, 2), (1, 1, 1), float(scale))
        key = (variant.permutation, variant.signs, float(variant.scale), variant.name)
        if key not in seen:
            variants.append(variant)
            seen.add(key)
    return variants


def _load_sequence_sensor_candidates(
    paths,
    *,
    voxel_size_m: float,
    min_cluster_points: int,
) -> dict[str, CandidateFrame]:
    by_sensor: dict[str, list[CandidateFrame]] = {}
    for path in paths.point_cloud_files:
        sensor = _sensor_from_path(path, sequence_root=paths.root)
        try:
            frame = load_point_cloud_file_as_candidates(
                path,
                source=sensor,
                sequence_id=paths.sequence_id,
                time_s=_point_cloud_frame_time_s(path, sequence_root=paths.root),
                voxel_size_m=voxel_size_m,
                min_points=min_cluster_points,
            )
        except Exception:
            continue
        if not frame.rows.empty:
            by_sensor.setdefault(sensor, []).append(frame)
    return {sensor: merge_candidate_frames(frames) for sensor, frames in by_sensor.items()}


def _sensor_from_path(path: Path, *, sequence_root: Path) -> str:
    try:
        parts = path.relative_to(sequence_root).parts
    except ValueError:
        parts = path.parts
    normalized = [part.lower().replace("-", "_").replace(" ", "_") for part in parts]
    for sensor in ("lidar_360", "livox_avia", "radar_enhance_pcl"):
        if sensor in normalized:
            return sensor
    for part in reversed(normalized[:-1]):
        if any(token in part for token in ("lidar", "livox", "radar", "pcl", "cloud", "point")):
            return part
    return Path(path).parent.name or "point_cloud"


def _audit_variant(
    truth: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    sequence_id: str,
    sensor: str,
    variant: AlignmentVariant,
    max_time_delta_s: float | None,
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
    return {
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
        "truth_frame_count": int(len(truth)),
        "candidate_count": int(len(candidates)),
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


def _transform_candidates(candidates: pd.DataFrame, variant: AlignmentVariant) -> pd.DataFrame:
    rows = candidates.copy()
    if rows.empty:
        for column in ("audit_x_m", "audit_y_m", "audit_z_m"):
            rows[column] = np.nan
        return rows
    xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    transformed = xyz[:, variant.permutation] * np.asarray(variant.signs, dtype=float)
    transformed *= float(variant.scale)
    rows["audit_x_m"] = transformed[:, 0]
    rows["audit_y_m"] = transformed[:, 1]
    rows["audit_z_m"] = transformed[:, 2]
    return rows


def _median_translation(
    truth: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    max_time_delta_s: float | None,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for _, truth_row in truth.sort_values("time_s").iterrows():
        candidates = _nearest_time_candidates(
            transformed,
            float(truth_row["time_s"]),
            max_time_delta_s=max_time_delta_s,
        )
        if candidates.empty:
            continue
        candidate_xyz = candidates[["audit_x_m", "audit_y_m", "audit_z_m"]].to_numpy(float)
        truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
        distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        best = int(np.argmin(distances))
        residuals.append(truth_xyz - candidate_xyz[best])
    if not residuals:
        return np.zeros(3, dtype=float)
    return np.nanmedian(np.vstack(residuals), axis=0).astype(float)


def _nearest_truth_distances(
    truth: pd.DataFrame,
    transformed: pd.DataFrame,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.full(len(truth), np.nan, dtype=float)
    time_deltas = np.full(len(truth), np.nan, dtype=float)
    matched = np.zeros(len(truth), dtype=bool)
    if transformed.empty:
        return distances, time_deltas, matched
    for idx, (_, truth_row) in enumerate(truth.sort_values("time_s").iterrows()):
        candidates = _nearest_time_candidates(
            transformed,
            float(truth_row["time_s"]),
            max_time_delta_s=max_time_delta_s,
        )
        if candidates.empty:
            continue
        candidate_xyz = candidates[["audit_x_m", "audit_y_m", "audit_z_m"]].to_numpy(float)
        truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
        candidate_distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        distances[idx] = float(np.min(candidate_distances))
        time_deltas[idx] = float(candidates["time_s"].iloc[0] - float(truth_row["time_s"]))
        matched[idx] = True
    return distances, time_deltas, matched


def _nearest_time_candidates(
    candidates: pd.DataFrame,
    truth_time_s: float,
    *,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    times = pd.to_numeric(candidates["time_s"], errors="coerce")
    finite = np.isfinite(times.to_numpy(float))
    if not finite.any():
        return candidates.iloc[0:0].copy()
    deltas = (times - float(truth_time_s)).abs()
    best_delta = float(deltas.loc[finite].min())
    if max_time_delta_s is not None and best_delta > float(max_time_delta_s):
        return candidates.iloc[0:0].copy()
    return candidates.loc[finite & (np.abs(deltas - best_delta) <= 1.0e-9)].copy()


def _finite_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = truth.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _finite_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    if rows.empty:
        return rows
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["time_s", "source"]).reset_index(drop=True)


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _threshold_fraction(distances: np.ndarray, threshold_m: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(np.sum(np.isfinite(distances) & (distances <= float(threshold_m))) / denominator)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else np.nan


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else np.nan


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
        "truth_frame_count",
        "candidate_count",
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


def _parse_max_time_delta(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "inf", "infinite", "unbounded"}:
        return None
    return float(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-coordinate-audit",
        description="audit MMUAD point-cloud coordinate frames and time alignment",
    )
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--truth-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sequence-glob", default="*")
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
        output_csv = args.output_dir / "mmuad_coordinate_alignment_audit.csv"
    scales = tuple(args.scales) if args.scales else (0.001, 0.01, 1.0)
    audit = build_coordinate_alignment_audit(
        args.sequence_root,
        args.truth_file,
        sequence_glob=args.sequence_glob,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
        max_time_delta_s=args.max_time_delta_s,
        include_translation_diagnostic=not args.skip_translation_diagnostic,
        scales=scales,
    )
    path = write_coordinate_alignment_audit(audit, output_csv)
    print("mmuad_coordinate_alignment_audit=ok")
    print(f"output_csv={path}")
    print(f"rows={len(audit)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
