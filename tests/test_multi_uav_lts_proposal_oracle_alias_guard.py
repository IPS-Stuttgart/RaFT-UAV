from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.proposal_oracle import audit_proposal_banks


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_derived_oracle_output_cannot_alias_truth_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    truth = output / "oracle_predictions" / "detector"
    proposals = tmp_path / "proposals"
    truth_path = truth / "S.txt"
    proposal_path = proposals / "S.txt"
    _write(truth_path, "1,1,0,0,10,10,1,1,1\n")
    _write(proposal_path, "1,1,0,0,10,10,0.5,1,1\n")
    original_truth = truth_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="must not alias the truth directory"):
        audit_proposal_banks(
            {"detector": proposals},
            truth,
            output,
            confidence_thresholds=(0.0,),
            iou_thresholds=(0.5,),
        )

    assert truth_path.read_text(encoding="utf-8") == original_truth


def test_derived_oracle_output_cannot_alias_proposal_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    truth = tmp_path / "truth"
    proposals = output / "oracle_predictions" / "detector"
    truth_path = truth / "S.txt"
    proposal_path = proposals / "S.txt"
    _write(truth_path, "1,1,0,0,10,10,1,1,1\n")
    _write(proposal_path, "1,1,0,0,10,10,0.5,1,1\n")
    original_proposal = proposal_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="must not alias proposal source"):
        audit_proposal_banks(
            {"detector": proposals},
            truth,
            output,
            confidence_thresholds=(0.0,),
            iou_thresholds=(0.5,),
        )

    assert proposal_path.read_text(encoding="utf-8") == original_proposal


def test_derived_fused_output_cannot_alias_truth_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    truth = output / "oracle_predictions" / "fused"
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    truth_path = truth / "S.txt"
    _write(truth_path, "1,1,0,0,10,10,1,1,1\n")
    _write(source_a / "S.txt", "1,1,0,0,10,10,0.5,1,1\n")
    _write(source_b / "S.txt", "1,1,0,0,10,10,0.4,1,1\n")
    original_truth = truth_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="source 'fused'.*truth directory"):
        audit_proposal_banks(
            {"source_a": source_a, "source_b": source_b},
            truth,
            output,
            confidence_thresholds=(0.0,),
            iou_thresholds=(0.5,),
        )

    assert truth_path.read_text(encoding="utf-8") == original_truth
