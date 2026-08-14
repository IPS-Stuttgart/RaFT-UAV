from __future__ import annotations

import json

import pytest

from raft_uav.mmuad.splits import load_split_manifest


def test_split_manifest_rejects_sequence_assigned_to_multiple_splits(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "sequences": [
                    {"sequence_id": "seq001", "split": "train"},
                    {"sequence_id": "seq001", "split": "val"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple splits") as exc_info:
        load_split_manifest(path)

    message = str(exc_info.value)
    assert "seq001" in message
    assert "train" in message
    assert "val" in message


def test_split_manifest_rejects_normalized_reference_overlap(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["seq001"], "val": ["./seq001/"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple splits"):
        load_split_manifest(path)


def test_split_manifest_allows_distinct_paths_with_same_basename(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"train": ["site_a/seq001"], "val": ["site_b/seq001"]}),
        encoding="utf-8",
    )

    assert load_split_manifest(path) == {
        "train": ("site_a/seq001",),
        "val": ("site_b/seq001",),
    }


def test_split_manifest_allows_window_view_of_same_partition(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"val": ["val/seq001"], "val_windows": ["val\\seq001"]}),
        encoding="utf-8",
    )

    assert load_split_manifest(path) == {
        "val": ("val/seq001",),
        "val_windows": ("val\\seq001",),
    }
