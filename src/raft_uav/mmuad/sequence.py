"""Sequence discovery/loading helpers for MMUAD-style directory exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from raft_uav.mmuad.calibration import (
    CalibrationSet,
    load_calibration_auto,
    transform_candidate_frame,
)
from raft_uav.mmuad.io import (
    load_candidate_csv,
    load_point_cloud_file_as_candidates,
    load_truth_csv,
    merge_candidate_frames,
)
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame


@dataclass(frozen=True)
class SequencePaths:
    """Paths discovered for one exported sequence."""

    sequence_id: str
    root: Path
    candidate_csvs: tuple[Path, ...]
    point_cloud_files: tuple[Path, ...]
    truth_csv: Path | None
    calibration_file: Path | None


def discover_sequence_paths(root: Path, *, sequence_glob: str = "*") -> list[SequencePaths]:
    """Discover sequence folders in an exported MMUAD-style directory.

    The helper intentionally supports normalized/exported files rather than the
    official raw archive.  It looks for common names such as ``candidates.csv``,
    ``*_candidates.csv``, ``points.csv``, ``*_points.csv``, ASCII ``*.pcd``, ASCII ``*.ply``, ``truth.csv``, and
    ``calibration.json`` under each sequence folder.  If ``root`` itself holds
    such files, it is treated as a single sequence.
    """

    root = Path(root)
    if _looks_like_sequence(root):
        return [_sequence_from_dir(root)]
    sequences = [
        _sequence_from_dir(path)
        for path in sorted(root.glob(sequence_glob))
        if path.is_dir() and _looks_like_sequence(path)
    ]
    return sequences


def load_sequence_export(
    paths: SequencePaths,
    *,
    apply_calibration: bool = True,
    voxel_size_m: float = 0.75,
    min_cluster_points: int = 3,
) -> tuple[CandidateFrame, TruthFrame | None, CalibrationSet | None]:
    """Load candidates/truth for one discovered sequence export."""

    candidate_frames = [
        load_candidate_csv(path, default_sequence_id=paths.sequence_id)
        for path in paths.candidate_csvs
    ]
    candidate_frames.extend(
        load_point_cloud_file_as_candidates(
            path,
            source=path.stem.replace("_points", "-cluster"),
            sequence_id=paths.sequence_id,
            voxel_size_m=voxel_size_m,
            min_points=min_cluster_points,
        )
        for path in paths.point_cloud_files
    )
    if not candidate_frames:
        raise ValueError(f"no candidate or point-cloud files discovered for {paths.root}")
    candidates = merge_candidate_frames(candidate_frames)
    truth = (
        load_truth_csv(paths.truth_csv, default_sequence_id=paths.sequence_id)
        if paths.truth_csv is not None
        else None
    )
    calibration = None
    if paths.calibration_file is not None:
        calibration = load_calibration_auto(paths.calibration_file)
        if apply_calibration:
            candidates = transform_candidate_frame(candidates, calibration)
    return candidates, truth, calibration


def _looks_like_sequence(path: Path) -> bool:
    if not path.is_dir():
        return False
    return bool(_candidate_files(path) or _point_files(path) or _truth_file(path))


def _sequence_from_dir(path: Path) -> SequencePaths:
    calibration = _first_existing(
        [
            path / "calibration.json",
            path / "calib.json",
            path / "extrinsics.json",
            path / "calibration.yaml",
            path / "calib.yaml",
            path / "extrinsics.yaml",
            path / "calibration.yml",
            path / "calib.yml",
            path / "extrinsics.yml",
            path / "calibration.txt",
            path / "extrinsics.txt",
        ]
    )
    return SequencePaths(
        sequence_id=path.name,
        root=path,
        candidate_csvs=tuple(_candidate_files(path)),
        point_cloud_files=tuple(_point_files(path)),
        truth_csv=_truth_file(path),
        calibration_file=calibration,
    )


def _candidate_files(path: Path) -> list[Path]:
    names = [path / "candidates.csv", path / "detections.csv"]
    files = [item for item in names if item.exists()]
    files.extend(sorted(path.glob("*_candidates.csv")))
    files.extend(sorted(path.glob("*_detections.csv")))
    return _unique_paths(files)


def _point_files(path: Path) -> list[Path]:
    names = [path / "points.csv", path / "point_cloud.csv", path / "lidar_points.csv"]
    files = [item for item in names if item.exists()]
    files.extend(sorted(path.glob("*_points.csv")))
    files.extend(sorted(path.glob("*.pcd")))
    files.extend(sorted(path.glob("*.ply")))
    files.extend(sorted(path.glob("*.npy")))
    files.extend(sorted(path.glob("*.npz")))
    return _unique_paths(files)


def _truth_file(path: Path) -> Path | None:
    return _first_existing(
        [
            path / "truth.csv",
            path / "ground_truth.csv",
            path / "gt.csv",
        ]
    )


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique
