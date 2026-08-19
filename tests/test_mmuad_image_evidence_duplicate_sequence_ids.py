from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.image_evidence import build_image_evidence
from raft_uav.mmuad.sequence import discover_sequence_paths


def test_image_evidence_rejects_duplicate_discovered_sequence_ids(tmp_path: Path) -> None:
    for wrapper in ("flight-a", "flight-b"):
        image_dir = tmp_path / wrapper / "seq001" / "Image"
        image_dir.mkdir(parents=True)
        (image_dir / "0.0.png").write_bytes(b"not-decoded-before-validation")

    sequences = discover_sequence_paths(tmp_path)
    assert [paths.sequence_id for paths in sequences] == ["seq001", "seq001"]

    with pytest.raises(ValueError, match=r"duplicate sequence_id.*seq001"):
        build_image_evidence(tmp_path)
