from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts.cli import score_lts_predictions


def test_lts_scorecard_maximizes_valid_iou_match_count(tmp_path: Path) -> None:
    truth_dir = tmp_path / "truth"
    prediction_dir = tmp_path / "predictions"
    truth_dir.mkdir()
    prediction_dir.mkdir()

    truth_dir.joinpath("S_00.txt").write_text(
        "1,1,0,0,2,1,1,1,1\n"
        "1,2,0,0,7,1,1,1,1\n",
        encoding="utf-8",
    )
    prediction_dir.joinpath("S_00.txt").write_text(
        "1,10,0,0,3,1,1,1,1\n"
        "1,11,1,0,2,1,1,1,1\n",
        encoding="utf-8",
    )

    scorecard = score_lts_predictions(
        prediction_dir,
        truth_dir,
        iou_threshold=0.3,
    )

    # The highest-IoU edge has IoU 2/3, but greedily taking it leaves only an
    # invalid 2/7 edge. The cardinality-optimal assignment uses 1/3 and 3/7.
    assert scorecard.matches == 2
    assert scorecard.false_positives == 0
    assert scorecard.false_negatives == 0
    assert scorecard.mota_like == 1.0
    assert scorecard.mean_matched_iou == pytest.approx(8.0 / 21.0)
