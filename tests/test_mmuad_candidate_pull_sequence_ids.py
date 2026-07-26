from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_pull import main as candidate_pull_main


def test_candidate_pull_cli_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    ).to_csv(results_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "ranker_score": [1.0],
            "confidence": [1.0],
            "cluster_point_count": [1],
        }
    ).to_csv(candidates_csv, index=False)

    assert candidate_pull_main(
        [
            "--results-in",
            str(results_csv),
            "--candidates",
            str(candidates_csv),
            "--results-out",
            str(output_csv),
            "--candidate-pull-policy",
            "constant",
            "--candidate-pull-alpha-xy",
            "1",
            "--candidate-pull-alpha-z",
            "1",
            "--candidate-pull-top-k",
            "1",
            "--candidate-pull-smoother",
            "none",
        ]
    ) == 0

    output = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    assert output["Sequence"].tolist() == ["001"]
    assert output["Position"].tolist() == ["(1,2,3)"]
