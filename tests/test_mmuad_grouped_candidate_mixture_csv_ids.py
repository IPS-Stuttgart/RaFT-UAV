from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map_grouped import main as grouped_mixture_main


def test_grouped_candidate_mixture_cli_preserves_initial_sequence_ids(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    initial_csv = tmp_path / "initial.csv"
    output_dir = tmp_path / "grouped_mixture"

    pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "source": "lidar",
                "track_id": "near",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "initialized",
                "candidate_branch": "translated",
                "x_m": 100.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    ).to_csv(candidates_csv, index=False)
    pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "state_x_m": 100.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            }
        ]
    ).to_csv(initial_csv, index=False)

    status = grouped_mixture_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-dir",
            str(output_dir),
            "--initial-estimates-csv",
            str(initial_csv),
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--top-k",
            "0",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(
        output_dir / "mmuad_candidate_mixture_estimates.csv",
        dtype={"sequence_id": str},
    )
    assert estimates["sequence_id"].tolist() == ["001"]
    assert estimates.loc[0, "state_x_m"] > 99.0
