from __future__ import annotations

import pandas as pd

from scripts import run_constrained_ablation_optimizer


def test_cli_preserves_multiple_constraints_on_the_same_column(tmp_path) -> None:
    summary_path = tmp_path / "summary.csv"
    output_path = tmp_path / "ranked.csv"
    pd.DataFrame(
        {
            "method": ["below", "within", "above"],
            "error_3d_rmse_m": [3.0, 2.0, 1.0],
            "truth_coverage_rate": [0.7, 0.9, 1.1],
        }
    ).to_csv(summary_path, index=False)

    result = run_constrained_ablation_optimizer.main(
        [
            str(summary_path),
            "--output-csv",
            str(output_path),
            "--constraint",
            "truth_coverage_rate:>=:0.8",
            "--constraint",
            "truth_coverage_rate:<=:1.0",
        ]
    )

    assert result == 0
    ranked = pd.read_csv(output_path)
    assert ranked.iloc[0]["method"] == "within"
    feasibility = ranked.set_index("method")["constraint_feasible"].to_dict()
    assert feasibility == {"within": True, "above": False, "below": False}
