from __future__ import annotations

from raft_uav.multi_uav_lts._records import parse_detection_text
from raft_uav.multi_uav_lts.proposal_bank_fusion import fuse_proposal_banks


def test_fusion_preserves_overlapping_alternatives_and_rekeys_per_frame(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "fused"
    first.mkdir()
    second.mkdir()
    (first / "BB1_00.txt").write_text(
        "1,1,10,10,8,8,0.9,1,1\n2,1,11,10,8,8,0.8,1,1\n",
        encoding="utf-8",
    )
    (second / "BB1_00.txt").write_text(
        "1,99,10.5,10,8,8,0.7,1,1\n2,44,30,30,6,6,0.6,1,1\n",
        encoding="utf-8",
    )

    summary = fuse_proposal_banks({"full": first, "tile": second}, output)
    rows = parse_detection_text(
        (output / "BB1_00.txt").read_text(encoding="utf-8"),
        source="fused",
    )

    assert summary.source_count == 2
    assert summary.sequence_count == 1
    assert summary.row_count == 4
    assert [(row.frame_id, row.object_id) for row in rows] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert [row.confidence for row in rows] == [0.9, 0.7, 0.8, 0.6]


def test_fusion_allows_complementary_sequence_coverage(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "fused"
    first.mkdir()
    second.mkdir()
    (first / "A_00.txt").write_text("1,1,0,0,2,2,1,1,1\n", encoding="utf-8")
    (second / "B_00.txt").write_text("1,1,1,1,2,2,1,1,1\n", encoding="utf-8")

    summary = fuse_proposal_banks({"a": first, "b": second}, output)

    assert summary.sequence_count == 2
    assert sorted(path.name for path in output.glob("*.txt")) == ["A_00.txt", "B_00.txt"]
