from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.mmuad.splits import load_split_manifest


def test_split_manifest_skips_string_placeholder_ids(tmp_path):
    path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq1", "nan", "None", "<NA>", "NaT", ""],
            "split": ["train", "train", "val", "test", "train", "val"],
        }
    ).to_csv(path, index=False)

    assert load_split_manifest(path) == {"train": ("seq1",)}


def test_split_manifest_skips_placeholder_split_labels(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "sequences": [
                    {"sequence_id": "seq1", "split": "train"},
                    {"sequence_id": "seq2", "split": "None"},
                    {"sequence_id": "seq3", "split": "<NA>"},
                    {"sequence_id": "seq4", "split": ""},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_split_manifest(path) == {"train": ("seq1",)}


def test_split_manifest_skips_boolean_sequence_ids(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "sequences": [
                    {"sequence_id": "seq1", "split": "train"},
                    {"sequence_id": True, "split": "train"},
                    {"sequence_id": False, "split": "val"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_split_manifest(path) == {"train": ("seq1",)}


def test_split_manifest_all_placeholder_csv_still_raises(tmp_path):
    path = tmp_path / "splits.csv"
    pd.DataFrame({"sequence_id": ["nan", "None"], "split": ["train", "val"]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="CSV split manifest"):
        load_split_manifest(path)
