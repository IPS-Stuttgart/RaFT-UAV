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


def merge_candidate_frames(frames: Iterable[CandidateFrame]) -> CandidateFrame:
    """Merge several candidate frames, preserving normalization."""

    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return CandidateFrame(normalize_candidate_columns(pd.DataFrame()))
    return CandidateFrame(normalize_candidate_columns(pd.concat(rows, ignore_index=True)))


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
