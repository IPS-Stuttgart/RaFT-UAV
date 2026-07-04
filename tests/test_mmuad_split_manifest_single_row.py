import json

from raft_uav.mmuad.splits import load_split_manifest


def test_json_split_manifest_accepts_single_row_object(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps(
            {
                "sequence_id": "seq001",
                "split": "train",
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {"train": ("seq001",)}


def test_yaml_split_manifest_accepts_single_row_alias_object(tmp_path):
    path = tmp_path / "split.yaml"
    path.write_text("id: seq002\nsubset: Val\n", encoding="utf-8")

    manifest = load_split_manifest(path)

    assert manifest == {"Val": ("seq002",)}
