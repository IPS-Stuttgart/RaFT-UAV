from __future__ import annotations

from pathlib import Path

from raft_uav.mmuad.layout import inspect_mmuad_layout


def test_layout_inspector_uses_leaf_sequence_under_nested_grouping(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "val" / "fog" / "seq_nested"
    livox = sequence / "livox_avia"
    truth = sequence / "ground_truth"
    livox.mkdir(parents=True)
    truth.mkdir()
    (sequence / "calibration.json").write_text("{}", encoding="utf-8")
    (livox / "1706255054.386069.npy").write_bytes(b"placeholder")
    (truth / "1706255054.386069.npy").write_bytes(b"placeholder")

    summary = inspect_mmuad_layout(tmp_path)

    candidates = summary["sequence_candidates"]
    assert [row["sequence_id"] for row in candidates] == ["seq_nested"]
    candidate = candidates[0]
    assert candidate["has_candidates_or_points"] is True
    assert candidate["has_truth_or_labels"] is True
    assert candidate["has_calibration"] is True
