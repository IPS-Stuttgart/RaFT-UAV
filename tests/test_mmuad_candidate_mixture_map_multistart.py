from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
)
from raft_uav.mmuad.candidate_mixture_map_multistart import (
    CandidateMixtureMultiStartConfig,
    build_candidate_mixture_initializations,
    compute_candidate_mixture_selection_objective,
    main as multistart_main,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.2,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"translated-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 8.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
                    "predicted_sigma_m": 8.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def test_branch_start_uses_branch_rows_and_global_fallback() -> None:
    rows = _candidate_rows()
    rows = rows.loc[~((rows["time_s"] == 2.0) & (rows["candidate_branch"] == "translated"))]
    starts = build_candidate_mixture_initializations(
        rows,
        mixture_config=CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
        ),
        multistart_config=CandidateMixtureMultiStartConfig(max_branch_starts=4),
    )

    translated = starts["branch:translated"]
    assert translated is not None
    translated = translated.sort_values("time_s").reset_index(drop=True)
    assert translated.loc[0, "state_x_m"] == 8.0
    assert translated.loc[1, "state_x_m"] == 9.0
    assert translated.loc[2, "state_x_m"] == 2.0


def test_selection_objective_uses_final_mixture_evidence_without_truth() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [0.0, 0.0],
        }
    )
    good = CandidateMixtureMapResult(
        estimates=estimates,
        assignments=pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": [0.0, 1.0],
                "mixture_log_weight": [0.0, 0.0],
            }
        ),
        iteration_summary=pd.DataFrame(),
        summary={},
    )
    poor = CandidateMixtureMapResult(
        estimates=estimates,
        assignments=pd.DataFrame(
            {
                "sequence_id": ["seqA", "seqA"],
                "time_s": [0.0, 1.0],
                "mixture_log_weight": [-2.0, -2.0],
            }
        ),
        iteration_summary=pd.DataFrame(),
        summary={},
    )
    config = CandidateMixtureMapConfig(smoothness_weight=0.0)

    good_objective = compute_candidate_mixture_selection_objective(
        good,
        mixture_config=config,
    )
    poor_objective = compute_candidate_mixture_selection_objective(
        poor,
        mixture_config=config,
    )

    assert good_objective["selection_objective"] == 0.0
    assert poor_objective["selection_objective"] == 4.0
    assert good_objective["selection_objective"] < poor_objective["selection_objective"]


def test_multistart_cli_writes_selected_result_and_restart_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = multistart_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "0",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--score-weight",
            "0",
            "--sigma-log-weight",
            "3",
            "--smoothness-weight",
            "100",
            "--iterations",
            "3",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_multistart_summary.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_multistart_initializations.csv").exists()
    summary_path = output_dir / "mmuad_candidate_mixture_multistart_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["truth_used_for_selection"] is False
    assert summary["start_count"] >= 4
    assert isinstance(summary["selected_start"], str)
    start_table = pd.read_csv(output_dir / "mmuad_candidate_mixture_multistart_summary.csv")
    assert int(start_table["selected"].sum()) == 1
    assert np.isfinite(start_table.loc[start_table["selected"], "selection_objective"]).all()
