from __future__ import annotations

import json

import pytest

from raft_uav.mmuad.splits import load_split_manifest


@pytest.mark.parametrize(
    "content",
    [
        "sequence_id, Sequence_ID ,split\nseq001,seq999,train\n",
        "sequence_id,split, SPLIT \nseq001,train,val\n",
    ],
)
def test_split_manifest_rejects_duplicate_normalized_csv_alias_headers(
    tmp_path,
    content,
):
    path = tmp_path / "splits.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous keys matching"):
        load_split_manifest(path)


def test_split_manifest_rejects_duplicate_normalized_json_container_keys(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "splits": {"train": ["seq001"]},
                " SPLITS ": {"val": ["seq002"]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous keys matching"):
        load_split_manifest(path)
