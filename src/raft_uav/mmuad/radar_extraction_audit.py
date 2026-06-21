"""Audit MMUAD radar point-cloud raw arrays versus clustered candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.coordinate_alignment_audit import _sensor_from_path
from raft_uav.mmuad.io import load_point_cloud_file_as_candidates
from raft_uav.mmuad.sequence import _point_cloud_frame_time_s, discover_sequence_paths


def build_radar_extraction_audit(
    sequence_root: Path,
    *,
    sequence_glob: str = "*",
    voxel_size_m: float = 0.75,
) -> pd.DataFrame:
    """Return per-frame radar raw-array and candidate-clustering diagnostics."""

    records: list[dict[str, Any]] = []
    for paths in discover_sequence_paths(Path(sequence_root), sequence_glob=sequence_glob):
        for path in paths.point_cloud_files:
            sensor = _sensor_from_path(path, sequence_root=paths.root)
            if sensor != "radar_enhance_pcl":
                continue
            timestamp = _point_cloud_frame_time_s(path, sequence_root=paths.root)
            raw = _raw_xyz_summary(path)
            cluster_counts = {
                min_points: _cluster_count(
                    path,
                    sequence_id=paths.sequence_id,
                    timestamp=timestamp,
                    voxel_size_m=voxel_size_m,
                    min_points=min_points,
                )
                for min_points in (1, 3, 5)
            }
            records.append(
                {
                    "sequence": paths.sequence_id,
                    "timestamp": float(timestamp) if timestamp is not None else np.nan,
                    "raw_shape": raw["raw_shape"],
                    "raw_point_count": raw["raw_point_count"],
                    "finite_xyz_count": raw["finite_xyz_count"],
                    "min_xyz": raw["min_xyz"],
                    "max_xyz": raw["max_xyz"],
                    "range_min": raw["range_min"],
                    "range_median": raw["range_median"],
                    "range_max": raw["range_max"],
                    "cluster_count_min1": cluster_counts[1],
                    "cluster_count_min3": cluster_counts[3],
                    "cluster_count_min5": cluster_counts[5],
                    "reason_no_candidates": _reason_no_candidates(raw, cluster_counts),
                }
            )
    return pd.DataFrame.from_records(records, columns=_audit_columns()).sort_values(
        ["sequence", "timestamp"],
        na_position="last",
    )


def write_radar_extraction_audit(frame: pd.DataFrame, path: Path) -> Path:
    """Write ``mmuad_radar_extraction_audit.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _raw_xyz_summary(path: Path) -> dict[str, Any]:
    try:
        array = _load_numpy_array(path)
        raw_shape = _shape_text(array)
        raw_point_count = _raw_point_count(array)
        xyz = _xyz_array(array)
    except Exception as exc:
        return {
            "raw_shape": "unreadable",
            "raw_point_count": 0,
            "finite_xyz_count": 0,
            "min_xyz": "",
            "max_xyz": "",
            "range_min": np.nan,
            "range_median": np.nan,
            "range_max": np.nan,
            "raw_error": str(exc),
        }
    finite_xyz = _finite_xyz(xyz)
    if finite_xyz.size == 0:
        return {
            "raw_shape": raw_shape,
            "raw_point_count": raw_point_count,
            "finite_xyz_count": 0,
            "min_xyz": "",
            "max_xyz": "",
            "range_min": np.nan,
            "range_median": np.nan,
            "range_max": np.nan,
            "raw_error": "",
        }
    ranges = np.linalg.norm(finite_xyz, axis=1)
    return {
        "raw_shape": raw_shape,
        "raw_point_count": raw_point_count,
        "finite_xyz_count": int(len(finite_xyz)),
        "min_xyz": _vector_text(np.min(finite_xyz, axis=0)),
        "max_xyz": _vector_text(np.max(finite_xyz, axis=0)),
        "range_min": float(np.min(ranges)),
        "range_median": float(np.median(ranges)),
        "range_max": float(np.max(ranges)),
        "raw_error": "",
    }


def _load_numpy_array(path: Path) -> np.ndarray:
    if Path(path).suffix.lower() not in {".npy", ".npz"}:
        raise ValueError(f"unsupported raw radar format {Path(path).suffix!r}")
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.lib.npyio.NpzFile):
        key = "points" if "points" in payload.files else payload.files[0]
        return np.asarray(payload[key])
    return np.asarray(payload)


def _shape_text(array: np.ndarray) -> str:
    if array.shape:
        return "x".join(str(int(item)) for item in array.shape)
    return "scalar"


def _raw_point_count(array: np.ndarray) -> int:
    if array.size == 0:
        return 0
    if array.dtype.names:
        return int(len(array))
    if array.ndim == 1:
        return 1 if array.shape[0] >= 3 else 0
    return int(array.shape[0])


def _xyz_array(array: np.ndarray) -> np.ndarray:
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    if array.dtype.names:
        names = {name.lower(): name for name in array.dtype.names}
        columns = []
        for canonical, aliases in {
            "x": ("x", "x_m", "point.x", "position.x"),
            "y": ("y", "y_m", "point.y", "position.y"),
            "z": ("z", "z_m", "point.z", "position.z"),
        }.items():
            source = next((names[alias] for alias in aliases if alias in names), None)
            if source is None:
                raise ValueError(f"structured array missing {canonical!r} coordinate field")
            columns.append(np.asarray(array[source], dtype=float))
        return np.column_stack(columns)
    if array.ndim == 1:
        if array.shape[0] < 3:
            raise ValueError(f"raw radar vector must contain at least 3 values, got {array.shape}")
        return np.asarray(array[:3], dtype=float).reshape(1, 3)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"raw radar array must be shape (N, >=3), got {array.shape}")
    return np.asarray(array[:, :3], dtype=float)


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.empty((0, 3), dtype=float)
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite]


def _vector_text(values: np.ndarray) -> str:
    return json.dumps([float(item) for item in values], separators=(",", ":"))


def _cluster_count(
    path: Path,
    *,
    sequence_id: str,
    timestamp: float | None,
    voxel_size_m: float,
    min_points: int,
) -> int:
    try:
        frame = load_point_cloud_file_as_candidates(
            path,
            source="radar_enhance_pcl",
            sequence_id=sequence_id,
            time_s=timestamp,
            voxel_size_m=voxel_size_m,
            min_points=min_points,
        )
    except Exception:
        return 0
    return int(len(frame.rows))


def _reason_no_candidates(raw: dict[str, Any], cluster_counts: dict[int, int]) -> str:
    if int(cluster_counts.get(3, 0)) > 0:
        return "candidates_present_min3"
    raw_error = str(raw.get("raw_error") or "")
    if raw_error:
        return "malformed_raw"
    if int(raw.get("raw_point_count") or 0) <= 0:
        return "raw_empty"
    if int(raw.get("finite_xyz_count") or 0) <= 0:
        return "no_finite_xyz"
    if int(cluster_counts.get(1, 0)) > 0:
        return "clusters_below_min3"
    return "finite_points_but_no_min1_clusters"


def _audit_columns() -> list[str]:
    return [
        "sequence",
        "timestamp",
        "raw_shape",
        "raw_point_count",
        "finite_xyz_count",
        "min_xyz",
        "max_xyz",
        "range_min",
        "range_median",
        "range_max",
        "cluster_count_min1",
        "cluster_count_min3",
        "cluster_count_min5",
        "reason_no_candidates",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-radar-extraction-audit",
        description="audit raw MMUAD radar point-cloud arrays versus clustered candidates",
    )
    parser.add_argument("sequence_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    args = parser.parse_args(argv)

    output_csv = args.output_csv
    if output_csv is None:
        if args.output_dir is None:
            raise SystemExit("provide --output-dir or --output-csv")
        output_csv = args.output_dir / "mmuad_radar_extraction_audit.csv"
    audit = build_radar_extraction_audit(
        args.sequence_root,
        sequence_glob=args.sequence_glob,
        voxel_size_m=args.voxel_size_m,
    )
    path = write_radar_extraction_audit(audit, output_csv)
    print("mmuad_radar_extraction_audit=ok")
    print(f"output_csv={path}")
    print(f"rows={len(audit)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
