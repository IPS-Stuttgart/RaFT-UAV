from __future__ import annotations

from raft_uav.multi_uav_lts._records import parse_detection_text, prepare_sequence


def test_prepare_sequence_materializes_one_separator_per_sparse_gap() -> None:
    rows = parse_detection_text(
        "1,1,0,0,10,10,1,1,1\n10000,1,1,0,10,10,1,1,1\n",
        source="sparse.txt",
    )

    prepared = prepare_sequence(rows, rows)

    assert prepared.frame_count == 10_000
    assert len(prepared.gt_ids) == 3
    assert len(prepared.tracker_ids) == 3
    assert len(prepared.similarity_scores) == 3
    assert [len(ids) for ids in prepared.gt_ids] == [1, 0, 1]
    assert [len(ids) for ids in prepared.tracker_ids] == [1, 0, 1]
    assert [matrix.shape for matrix in prepared.similarity_scores] == [
        (1, 1),
        (0, 0),
        (1, 1),
    ]
