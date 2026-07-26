from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import main


def test_candidate_oracle_attribution_cli_preserves_truth_sequence_ids(
    tmp_path: Path,
) -> None:
    candidate_csv = tmp_path / "candidate.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    candidate_csv.write_text(
        "sequence_id,time_s,source,track_id,x_m,y_m,z_m,"
        "candidate_reservoir_score\n"
        "001,0.0,lidar,track-1,1.0,2.0,3.0,1.0\n",
        encoding="utf-8",
    )
    truth_csv.write_text(
        "sequence_id,time_s,x_m,y_m,z_m\n"
        "001,0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--candidate",
            f"raw={candidate_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
        ]
    ) == 0

    frames = pd.read_csv(
        output_dir / "mmuad_candidate_oracle_attribution_frames.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert frames["sequence_id"].tolist() == ["001"]
