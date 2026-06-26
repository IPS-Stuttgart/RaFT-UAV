from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_branch_balance import (
    BranchBalanceConfig,
    main as branch_balance_main,
    prepare_branch_balanced_candidates,
    run_branch_balanced_candidate_mixture_map,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


def _branch_candidates() -> pd.DataFrame:
    records = []
    for time_s in range(5):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-a-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s + 20.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.99,
                    "predicted_sigma_m": 20.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-b-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s + 21.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.98,
                    "predicted_sigma_m": 20.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"calibrated-{time_s}",
                    "candidate_branch": "source_translation_calibrated",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.01,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 5,
            "time_s": np.arange(5, dtype=float),
            "x_m": np.arange(5, dtype=float),
            "y_m": np.zeros(5),
            "z_m": np.ones(5),
        }
    )


def test_round_robin_keeps_low_global_score_branch() -> None:
    global_rows = prepare_branch_balanced_candidates(
        _branch_candidates(),
        config=BranchBalanceConfig(
            top_k=2,
            score_column="ranker_score",
            global_score_blend=1.0,
            selection_mode="global",
        ),
    )
    round_robin_rows = prepare_branch_balanced_candidates(
        _branch_candidates(),
        config=BranchBalanceConfig(
            top_k=2,
            score_column="ranker_score",
            global_score_blend=1.0,
            selection_mode="round-robin",
        ),
    )

    assert set(global_rows["candidate_branch"]) == {"raw"}
    assert set(round_robin_rows["candidate_branch"]) == {
        "raw",
        "source_translation_calibrated",
    }
    assert round_robin_rows.groupby(["sequence_id", "time_s"]).size().eq(2).all()


def test_branch_rank_normalization_is_comparable_across_score_scales() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["s"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar"] * 4,
            "track_id": ["raw-a", "raw-b", "cal-a", "cal-b"],
            "candidate_branch": ["raw", "raw", "calibrated", "calibrated"],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0] * 4,
            "z_m": [0.0] * 4,
            "ranker_score": [100.0, 90.0, 0.9, 0.8],
        }
    )

    balanced = prepare_branch_balanced_candidates(
        rows,
        config=BranchBalanceConfig(
            top_k=4,
            score_column="ranker_score",
            branch_score_normalization="rank",
            global_score_blend=0.0,
        ),
    )

    top_by_branch = balanced.sort_values("branch_balance_branch_rank").groupby(
        "candidate_branch",
        sort=True,
    ).first()
    assert set(top_by_branch["branch_balance_branch_score"]) == {1.0}


def test_branch_balanced_mixture_recovers_low_score_calibrated_trajectory() -> None:
    result = run_branch_balanced_candidate_mixture_map(
        _branch_candidates(),
        branch_config=BranchBalanceConfig(
            top_k=2,
            score_column="ranker_score",
            global_score_blend=1.0,
            selection_mode="round-robin",
        ),
        mixture_config=CandidateMixtureMapConfig(
            sigma_column="predicted_sigma_m",
            smoothness_weight=100.0,
            iterations=5,
        ),
        truth=_truth_rows(),
    )

    assert result.mixture_result.summary["metrics"]["pooled"]["rmse_3d_m"] < 0.05
    dominant = result.mixture_result.assignments.loc[
        result.mixture_result.assignments["mixture_dominant"]
    ]
    assert set(dominant["candidate_branch"]) == {"source_translation_calibrated"}
    assert result.summary["balanced_branch_counts"] == {
        "raw": 5,
        "source_translation_calibrated": 5,
    }


def test_branch_balanced_mixture_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _branch_candidates().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = branch_balance_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--score-column",
            "ranker_score",
            "--global-score-blend",
            "1.0",
            "--smoothness-weight",
            "100",
            "--iterations",
            "5",
        ]
    )

    assert status == 0
    for name in (
        "mmuad_branch_balanced_candidates.csv",
        "mmuad_branch_balanced_mixture_summary.json",
        "mmuad_candidate_mixture_estimates.csv",
        "mmuad_candidate_mixture_assignments.csv",
        "mmuad_candidate_mixture_iterations.csv",
        "mmuad_candidate_mixture_summary.json",
    ):
        assert (output_dir / name).exists()
    summary = json.loads(
        (output_dir / "mmuad_branch_balanced_mixture_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["branch_config"]["selection_mode"] == "round-robin"
    assert summary["mixture_summary"]["metrics"]["pooled"]["rmse_3d_m"] < 0.05


def test_branch_balanced_mixture_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-branch-balanced-mixture-map"]
        == "raft_uav.mmuad.candidate_mixture_branch_balance:main"
    )
