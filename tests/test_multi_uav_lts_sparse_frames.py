from __future__ import annotations

from raft_uav.multi_uav_lts._clear_identity import evaluate_clear
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


def test_sparse_gap_resets_clear_previous_timestep_continuity() -> None:
    truth = parse_detection_text(
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,2,0,10,10,1,1,1\n"
        "3,1,0,0,10,10,1,1,1\n"
        "3,2,2,0,10,10,1,1,1\n",
        source="truth.txt",
    )
    predictions = parse_detection_text(
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,2,0,10,10,1,1,1\n"
        "3,1,2,0,10,10,1,1,1\n"
        "3,2,0,0,10,10,1,1,1\n",
        source="predictions.txt",
    )

    prepared = prepare_sequence(truth, predictions)
    clear = evaluate_clear(prepared, threshold=0.5)

    assert clear.tp == 4
    assert clear.fp == 0
    assert clear.fn == 0
    assert clear.id_switches == 2
    assert clear.motp_sum == 4.0
