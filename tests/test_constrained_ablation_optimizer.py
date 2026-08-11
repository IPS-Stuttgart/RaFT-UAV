from __future__ import annotations

import pandas as pd

from scripts.run_constrained_ablation_optimizer import main


def test_pareto_front_excludes_infeasible_configurations(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    output_csv = tmp_path / "ranked.csv"
    pd.DataFrame(
        [
            {"method": "feasible", "rmse": 2.0, "coverage": 0.95},
            {"method": "infeasible_but_lower_error", "rmse": 1.0, "coverage": 0.50},
        ]
    ).to_csv(summary_csv, index=False)

    assert (
        main(
            [
                str(summary_csv),
                "--output-csv",
                str(output_csv),
                "--objective",
                "rmse",
                "--constraint",
                "coverage:>=:0.9",
                "--pareto-minimize",
                "rmse",
            ]
        )
        == 0
    )

    ranked = pd.read_csv(output_csv).set_index("method")
    assert bool(ranked.loc["feasible", "constraint_feasible"])
    assert bool(ranked.loc["feasible", "pareto_front"])
    assert not bool(ranked.loc["infeasible_but_lower_error", "constraint_feasible"])
    assert not bool(ranked.loc["infeasible_but_lower_error", "pareto_front"])
