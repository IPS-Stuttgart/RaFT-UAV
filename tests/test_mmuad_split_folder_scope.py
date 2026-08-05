from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.sequence import SequencePaths
from raft_uav.mmuad.splits import filter_sequences_by_split_folder


def _sequence(root: Path, *, sequence_id: str) -> SequencePaths:
    return SequencePaths(
        sequence_id=sequence_id,
        root=root,
        candidate_csvs=(),
        candidate_trajectory_files=(),
        radar_polar_csvs=(),
        camera_detection_csvs=(),
        point_cloud_files=(),
        topic_map_jsons=(),
        truth_file=None,
        truth_files=(),
        class_files=(),
        calibration_file=None,
    )


def test_split_folder_filter_rejects_matching_directory_outside_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    in_scope_root = dataset_root / "train" / "seq001"
    out_of_scope_root = tmp_path / "other" / "train"
    in_scope_root.mkdir(parents=True)
    out_of_scope_root.mkdir(parents=True)

    in_scope = _sequence(in_scope_root, sequence_id="seq001")
    out_of_scope = _sequence(out_of_scope_root, sequence_id="unrelated")

    selected = filter_sequences_by_split_folder(
        [in_scope, out_of_scope],
        dataset_root,
        "train",
    )

    assert selected == [in_scope]


def test_split_folder_filter_accepts_root_when_root_is_named_split(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    split_root.mkdir()
    sequence = _sequence(split_root, sequence_id="train")

    selected = filter_sequences_by_split_folder([sequence], split_root, "TRAIN")

    assert selected == [sequence]
