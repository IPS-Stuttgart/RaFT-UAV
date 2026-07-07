from __future__ import annotations

from raft_uav.mmuad.splits import load_split_manifest, resolve_split_name


def test_split_manifest_accepts_padded_csv_alias_headers(tmp_path):
    path = tmp_path / "splits.csv"
    path.write_text(" Sequence , Subset \n001, Train \n002, Val \n", encoding="utf-8")

    manifest = load_split_manifest(path)

    assert manifest == {"Train": ("001",), "Val": ("002",)}
    assert resolve_split_name(manifest, " train ") == "Train"
