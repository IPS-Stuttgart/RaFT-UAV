from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_scorecard_compare import main


def _write_pose_table(path: Path, sequence_id: str, mse: float) -> None:
    path.write_text(
        "sequence_id,count,mse,rmse,mean_3d,median_3d,p95_3d,max_3d\n"
        f"{sequence_id},10,{mse},1.0,1.0,1.0,1.0,1.0\n",
        encoding="utf-8",
    )


def test_scorecard_compare_cli_preserves_zero_padded_sequence_ids(
    tmp_path: Path,
) -> None:
    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"
    delta_csv = tmp_path / "delta.csv"
    summary_json = tmp_path / "summary.json"
    _write_pose_table(baseline_csv, "001", 1.0)
    _write_pose_table(candidate_csv, "1", 2.0)

    assert (
        main(
            [
                "--baseline-pose-by-sequence-csv",
                str(baseline_csv),
                "--candidate-pose-by-sequence-csv",
                str(candidate_csv),
                "--pose-delta-csv",
                str(delta_csv),
                "--pose-delta-summary-json",
                str(summary_json),
            ]
        )
        == 0
    )

    delta = pd.read_csv(delta_csv, dtype={"sequence_id": "string"})
    assert sorted(delta["sequence_id"].tolist()) == ["001", "1"]
    assert not delta["matched_in_both"].any()

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["common_sequence_count"] == 0
    assert summary["candidate_only_sequence_count"] == 1
    assert summary["baseline_only_sequence_count"] == 1
