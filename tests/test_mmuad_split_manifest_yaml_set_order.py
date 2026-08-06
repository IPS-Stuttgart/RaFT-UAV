from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from raft_uav.mmuad.splits import load_split_manifest


class _ReverseSequenceSet(set[str]):
    def __iter__(self):
        return iter(("seq_c", "seq_a", "seq_b"))


def test_yaml_set_split_entries_have_deterministic_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "splits.yaml"
    path.write_text("train: !!set {}\n", encoding="utf-8")
    values = _ReverseSequenceSet({"seq_a", "seq_b", "seq_c"})
    monkeypatch.setattr(yaml, "safe_load", lambda _text: {"train": values})

    manifest = load_split_manifest(path)

    assert manifest == {"train": ("seq_a", "seq_b", "seq_c")}
