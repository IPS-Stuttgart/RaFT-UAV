from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.sequence import SequencePaths
from raft_uav.mmuad.splits import (
    filter_sequences_by_split,
    load_split_manifest,
    resolve_split_name,
    split_manifest_summary,
)


def _sequence(root: Path, sequence_id: str) -> SequencePaths:
    return SequencePaths(
        sequence_id=sequence_id,
        root=root / sequence_id,
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


def test_split_manifest_accepts_case_insensitive_split_container_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "Splits": {
                    "Train": {
                        "Sequences": [
                            {"Sequence_ID": "seq_train_a"},
                            {"ID": "seq_train_b"},
                        ]
                    },
                    "Val": {"Sequence_IDs": ["seq_val_a", "seq_val_b"]},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {
        "Train": ("seq_train_a", "seq_train_b"),
        "Val": ("seq_val_a", "seq_val_b"),
    }


def test_split_manifest_accepts_case_insensitive_top_level_sequence_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "Sequences": [
                    {"Sequence": "seq_train", "Subset": "train"},
                    {"ID": "seq_test", "Fold": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {"train": ("seq_train",), "test": ("seq_test",)}


def test_split_manifest_filter_resolves_split_name_case_insensitively(
    tmp_path: Path,
) -> None:
    sequences = [
        _sequence(tmp_path, "seq_train_a"),
        _sequence(tmp_path, "seq_train_b"),
        _sequence(tmp_path, "seq_val_a"),
    ]
    manifest = {
        "Train": ("seq_train_a", "seq_train_b"),
        "Val": ("seq_val_a",),
    }

    selected = filter_sequences_by_split(sequences, manifest, "train")

    assert [sequence.sequence_id for sequence in selected] == [
        "seq_train_a",
        "seq_train_b",
    ]


def test_split_manifest_summary_resolves_split_name_case_insensitively() -> None:
    manifest = {"Train": ("seq_train_a", "seq_train_b")}

    summary = split_manifest_summary(manifest)

    assert resolve_split_name(manifest, "train") == "Train"
    assert summary["train"] == {
        "count": 2,
        "sequence_ids": ["seq_train_a", "seq_train_b"],
    }
