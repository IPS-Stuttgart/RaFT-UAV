from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map_text_cli import main as mixture_text_main


def test_candidate_mixture_text_cli_preserves_zero_padded_template_sequence_ids(tmp_path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    template_csv = tmp_path / "template.csv"
    estimates_csv = tmp_path / "estimates.csv"

    pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "track-0",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "001",
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "track-1",
                "candidate_branch": "raw",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    ).to_csv(candidates_csv, index=False)
    template_csv.write_text(
        "Sequence,Timestamp,Position,Classification\n"
        "001,0.0,,\n"
        "001,1.0,,\n",
        encoding="utf-8",
    )

    status = mixture_text_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--target-template-csv",
            str(template_csv),
            "--output-estimates-csv",
            str(estimates_csv),
            "--top-k",
            "1",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(estimates_csv, dtype={"sequence_id": str})
    assert estimates["sequence_id"].tolist() == ["001", "001"]
    assert estimates["time_s"].astype(float).tolist() == [0.0, 1.0]
