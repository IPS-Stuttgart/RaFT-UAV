from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs, main


def _write_pose_csv(path: Path) -> None:
    path.write_text(
        "sequence_id,time_s,x_m,y_m,z_m,confidence\n"
        "001,0.0,1.0,2.0,3.0,0.9\n",
        encoding="utf-8",
    )


def test_candidate_reservoir_loader_preserves_zero_padded_sequence(
    tmp_path: Path,
) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    _write_pose_csv(candidate_csv)

    rows = load_candidate_inputs([f"raw={candidate_csv}"])

    assert rows.loc[0, "sequence_id"] == "001"
    assert rows.loc[0, "candidate_branch"] == "raw"


def test_candidate_reservoir_cli_oracle_preserves_zero_padded_sequence(
    tmp_path: Path,
) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    _write_pose_csv(candidate_csv)
    _write_pose_csv(truth_csv)

    assert (
        main(
            [
                "--candidate",
                f"raw={candidate_csv}",
                "--output-csv",
                str(output_csv),
                "--truth-csv",
                str(truth_csv),
                "--oracle-frame-csv",
                str(oracle_frame_csv),
            ]
        )
        == 0
    )

    reservoir = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    oracle_frames = pd.read_csv(oracle_frame_csv, dtype=str, keep_default_na=False)
    assert reservoir.loc[0, "sequence_id"] == "001"
    assert oracle_frames.loc[0, "sequence_id"] == "001"
