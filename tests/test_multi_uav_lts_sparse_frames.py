from __future__ import annotations

from raft_uav.multi_uav_lts._records import parse_detection_text, prepare_sequence


def test_prepare_sequence_materializes_only_observed_frames() -> None:
    rows = parse_detection_text(
        "1,1,0,0,10,10,1,1,1\n10000,1,1,0,10,10,1,1,1\n",
        source="sparse.txt",
    )

    prepared = prepare_sequence(rows, rows)

    assert prepared.frame_count == 10_000
    assert len(prepared.gt_ids) == 2
    assert len(prepared.tracker_ids) == 2
    assert len(prepared.similarity_scores) == 2
    assert all(matrix.shape == (1, 1) for matrix in prepared.similarity_scores)
