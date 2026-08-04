from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.proposal_oracle import audit_proposal_banks


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_fused_proposals_raise_recall_and_materialize_seeded_oracle(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "truth"
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    output = tmp_path / "output"
    _write(
        truth / "S.txt",
        "1,7,0,0,10,10,1,1,1\n"
        "1,9,100,0,10,10,1,1,1\n"
        "2,7,1,0,10,10,1,1,1\n"
        "2,9,99,0,10,10,1,1,1\n",
    )
    _write(source_a / "S.txt", "2,1,1,0,10,10,0.8,1,1\n")
    _write(source_b / "S.txt", "2,1,99,0,10,10,0.7,1,1\n")

    summary = audit_proposal_banks(
        {"source_a": source_a, "source_b": source_b},
        truth,
        output,
        confidence_thresholds=(0.0,),
        iou_thresholds=(0.5,),
        oracle_confidence_threshold=0.0,
        oracle_iou_threshold=0.5,
    )

    coverage = {
        row.source: (row.matched_count, row.truth_count)
        for row in summary.coverage
    }
    assert coverage == {
        "source_a": (1, 4),
        "source_b": (1, 4),
        "fused": (2, 4),
    }
    assert (
        output / "oracle_predictions" / "fused" / "S.txt"
    ).read_text(encoding="utf-8").splitlines() == [
        "1,7,0,0,10,10,1,1,1",
        "1,9,100,0,10,10,1,1,1",
        "2,7,1,0,10,10,0.8,1,1",
        "2,9,99,0,10,10,0.7,1,1",
    ]
    assert (output / "proposal_oracle_summary.json").is_file()
    assert (output / "coverage.csv").is_file()
    assert summary.oracle_scores[-1].source == "fused"


def test_one_proposal_cannot_cover_two_truth_objects(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    proposals = tmp_path / "proposals"
    _write(
        truth / "S.txt",
        "1,1,0,0,10,10,1,1,1\n"
        "1,2,4,0,10,10,1,1,1\n",
    )
    _write(proposals / "S.txt", "1,1,2,0,10,10,1,1,1\n")

    summary = audit_proposal_banks(
        {"detector": proposals},
        truth,
        tmp_path / "output",
        confidence_thresholds=(0.0,),
        iou_thresholds=(0.1,),
        oracle_confidence_threshold=0.0,
        oracle_iou_threshold=0.1,
    )

    assert summary.coverage[0].matched_count == 1
    assert summary.coverage[0].truth_count == 2


def test_invalid_proposal_confidence_is_rejected(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    proposals = tmp_path / "proposals"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(proposals / "S.txt", "1,1,0,0,10,10,1.1,1,1\n")

    with pytest.raises(ValueError, match="proposal confidence"):
        audit_proposal_banks(
            {"detector": proposals},
            truth,
            tmp_path / "output",
        )


def test_reserved_or_path_like_source_names_are_rejected(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    proposals = tmp_path / "proposals"
    _write(truth / "S.txt", "1,1,0,0,10,10,1,1,1\n")
    _write(proposals / "S.txt", "")

    with pytest.raises(ValueError, match="safe non-reserved"):
        audit_proposal_banks(
            {"..": proposals},
            truth,
            tmp_path / "output",
        )
