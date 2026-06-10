"""I/O helpers for normalized MMUAD tracking candidates."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import (
    CandidateFrame,
    TruthFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)


def load_candidate_csv(path: Path) -> CandidateFrame:
    """Load a normalized or alias-compatible candidate CSV."""

    rows = normalize_candidate_columns(pd.read_csv(path))
    frame = CandidateFrame(rows)
    frame.validate()
    return frame


def load_truth_csv(path: Path) -> TruthFrame:
    """Load a normalized or alias-compatible truth CSV."""

    rows = normalize_truth_columns(pd.read_csv(path))
    frame = TruthFrame(rows)
    frame.validate()
    return frame


def load_point_cloud_csv_as_candidates(
    path: Path,
    *,
    source: str = "lidar-cluster",
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    min_confidence: float = 0.0,
) -> CandidateFrame:
    """Cluster simple point-cloud CSV rows into UAV candidate centroids.

    Expected point columns are compatible with the normalized schema aliases:
    ``sequence_id``, ``time_s``, ``x_m``, ``y_m``, ``z_m``.  The clustering is a
    lightweight connected-component pass in voxel space, intended as a first
    baseline and smoke-test feature rather than a competitive MMUAD detector.
    """

    points = normalize_truth_columns(pd.read_csv(path))
    return _point_rows_to_candidates(
        points,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=min_confidence,
    )




def load_point_cloud_file_as_candidates(
    path: Path,
    *,
    source: str | None = None,
    sequence_id: str | None = None,
    time_s: float | None = None,
    voxel_size_m: float = 0.75,
    min_points: int = 3,
    min_confidence: float = 0.0,
) -> CandidateFrame:
    """Load CSV/PCD/PLY point-cloud files and cluster them into candidates.

    ASCII PCD and PLY are supported as a pragmatic exported-data bridge.  This
    is **not** a native Livox packet reader.  Files without per-point timestamps
    are treated as one frame; ``time_s`` is inferred from the filename when it
    contains a numeric token, otherwise it defaults to ``0.0``.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    source = source or path.stem.replace("_points", "-cluster")
    if suffix == ".csv":
        return load_point_cloud_csv_as_candidates(
            path,
            source=source,
            voxel_size_m=voxel_size_m,
            min_points=min_points,
            min_confidence=min_confidence,
        )
    if suffix == ".pcd":
        points = _read_ascii_pcd(path)
    elif suffix == ".ply":
        points = _read_ascii_ply(path)
    else:
        raise ValueError(f"unsupported point-cloud extension: {path.suffix}")
    sequence_id = sequence_id or path.parent.name
    if time_s is None:
        time_s = infer_time_s_from_filename(path)
    points["sequence_id"] = str(sequence_id)
    points["time_s"] = float(time_s)
    return _point_rows_to_candidates(
        points,
        source=source,
        voxel_size_m=voxel_size_m,
        min_points=min_points,
        min_confidence=min_confidence,
    )


def infer_time_s_from_filename(path: Path) -> float:
    """Infer a frame timestamp from the last numeric token in a filename."""

    import re

    tokens = re.findall(r"[-+]?\d*\.?\d+", Path(path).stem)
    if not tokens:
        return 0.0
    return float(tokens[-1])


def merge_candidate_frames(frames: Iterable[CandidateFrame]) -> CandidateFrame:
    """Merge several candidate frames, preserving normalization."""

    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return CandidateFrame(normalize_candidate_columns(pd.DataFrame()))
    return CandidateFrame(normalize_candidate_columns(pd.concat(rows, ignore_index=True)))




def _point_rows_to_candidates(
    points: pd.DataFrame,
    *,
    source: str,
    voxel_size_m: float,
    min_points: int,
    min_confidence: float,
) -> CandidateFrame:
    records: list[dict[str, object]] = []
    for (sequence_id, time_s), group in points.groupby(["sequence_id", "time_s"], sort=True):
        xyz = group[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
        for cluster_idx, members in enumerate(
            _voxel_connected_components(xyz, voxel_size_m=voxel_size_m, min_points=min_points)
        ):
            cluster = xyz[members]
            confidence = float(len(cluster))
            if confidence < float(min_confidence):
                continue
            centroid = cluster.mean(axis=0)
            spread = np.maximum(cluster.std(axis=0), 0.25)
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(time_s),
                    "source": source,
                    "track_id": f"{source}:{sequence_id}:{time_s}:{cluster_idx}",
                    "x_m": centroid[0],
                    "y_m": centroid[1],
                    "z_m": centroid[2],
                    "std_xy_m": float(max(spread[0], spread[1], 0.5)),
                    "std_z_m": float(max(spread[2], 0.5)),
                    "confidence": confidence,
                    "class_name": "uav",
                }
            )
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(records)))


def _read_ascii_pcd(path: Path) -> pd.DataFrame:
    fields: list[str] = []
    data_start = None
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        upper = stripped.upper()
        if upper.startswith("FIELDS"):
            fields = stripped.split()[1:]
        if upper.startswith("DATA"):
            if "ascii" not in upper.lower():
                raise ValueError("only ASCII PCD files are supported")
            data_start = idx + 1
            break
    if data_start is None or not fields:
        raise ValueError(f"invalid ASCII PCD file: {path}")
    rows = []
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < len(fields):
            continue
        rows.append({field: value for field, value in zip(fields, parts, strict=False)})
    return _normalize_point_frame(pd.DataFrame.from_records(rows), path=path)


def _read_ascii_ply(path: Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"invalid PLY file: {path}")
    vertex_count = 0
    properties: list[str] = []
    data_start = None
    in_vertex = False
    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.strip()
        if stripped.startswith("format") and "ascii" not in stripped:
            raise ValueError("only ASCII PLY files are supported")
        if stripped.startswith("element vertex"):
            vertex_count = int(stripped.split()[-1])
            in_vertex = True
            continue
        if stripped.startswith("element") and not stripped.startswith("element vertex"):
            in_vertex = False
        if in_vertex and stripped.startswith("property"):
            properties.append(stripped.split()[-1])
        if stripped == "end_header":
            data_start = idx + 1
            break
    if data_start is None or vertex_count <= 0 or not properties:
        raise ValueError(f"invalid ASCII PLY file: {path}")
    rows = []
    for line in lines[data_start : data_start + vertex_count]:
        parts = line.split()
        if len(parts) < len(properties):
            continue
        rows.append({field: value for field, value in zip(properties, parts, strict=False)})
    return _normalize_point_frame(pd.DataFrame.from_records(rows), path=path)


def _normalize_point_frame(frame: pd.DataFrame, *, path: Path) -> pd.DataFrame:
    aliases = {
        "x": "x_m",
        "y": "y_m",
        "z": "z_m",
        "X": "x_m",
        "Y": "y_m",
        "Z": "z_m",
    }
    out = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    missing = {"x_m", "y_m", "z_m"}.difference(out.columns)
    if missing:
        raise ValueError(f"point cloud {path} missing coordinate columns: {sorted(missing)}")
    for col in ("x_m", "y_m", "z_m"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.loc[np.isfinite(out[["x_m", "y_m", "z_m"]]).all(axis=1)].copy()


def _voxel_connected_components(
    xyz: np.ndarray,
    *,
    voxel_size_m: float,
    min_points: int,
) -> list[np.ndarray]:
    if xyz.size == 0:
        return []
    voxels = np.floor(xyz / float(voxel_size_m)).astype(int)
    voxel_to_points: dict[tuple[int, int, int], list[int]] = {}
    for idx, voxel in enumerate(voxels):
        voxel_to_points.setdefault(tuple(int(v) for v in voxel), []).append(idx)
    visited: set[tuple[int, int, int]] = set()
    components: list[np.ndarray] = []
    for voxel in voxel_to_points:
        if voxel in visited:
            continue
        queue: deque[tuple[int, int, int]] = deque([voxel])
        visited.add(voxel)
        point_indices: list[int] = []
        while queue:
            current = queue.popleft()
            point_indices.extend(voxel_to_points[current])
            for neighbor in _neighbors26(current):
                if neighbor in voxel_to_points and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(point_indices) >= int(min_points):
            components.append(np.asarray(point_indices, dtype=int))
    return components


def _neighbors26(voxel: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    x, y, z = voxel
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                yield (x + dx, y + dy, z + dz)
