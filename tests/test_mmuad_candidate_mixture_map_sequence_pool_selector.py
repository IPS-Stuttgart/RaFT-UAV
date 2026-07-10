from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_sequence_pool_selector import (
    CandidatePoolSequenceSelectorConfig,
    main as sequence_pool_main,
    run_sequence_pool_selector,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for sequence_id, precise_branch in (("seqA", "raw"), ("seqB", "translated")):
        for time_s in range(5):
            for branch in ("raw", "translated"):
                precise = branch == precise_branch
                records.append(
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": branch,
                        "track_id": f"{sequence_id}-{branch}-{time_s}",
                        "candidate_branch": branch,
                        "x_m": float(time_s if precise else time_s + 100.0),
                        "y_m": 0.0,
                        "z_m": 1.0,
                        "ranker_score": 0.5,
                        "predicted_sigma_m": 1.0 if precise else 20.0,
                    }
                )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    records = []
    for sequence_id in ("seqA", "seqB"):
        for time_s in range(5):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(time_s),
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        sigma_column="predicted_sigma_m",
        score_weight=0.0,
        sigma_log_weight=3.0,
        smoothness_weight=100.0,
        iterations=4,
    )


def test_sequence_pool_selector_removes_different_harmful_branches() -> None:
    result = run_sequence_pool_selector(
        _candidate_rows(),
        mixture_config=_mixture_config(),
        selector_config=CandidatePoolSequenceSelectorConfig(
            group_column="candidate_branch",
            max_leave_one_out=4,
            min_group_frame_fraction=0.0,
        ),
        truth=_truth_rows(),
    )

    assert result.selected_pool_by_sequence == {
        "seqA": "without_candidate_branch_translated",
        "seqB": "without_candidate_branch_raw",
    }
    selected = result.pool_summary.loc[result.pool_summary["selected"]]
    assert selected["component_count_penalty"].eq(0.0).all()
    full = result.pool_summary.loc[result.pool_summary["pool_label"] == "full_pool"]
    assert np.allclose(full["component_count_penalty"], 5.0 * np.log(2.0))
    assert result.selected_result.summary["metrics"]["pooled"]["rmse_3d_m"] < 1.0e-6
    assert set(result.selected_candidates["selected_candidate_pool"]) == {
        "without_candidate_branch_translated",
        "without_candidate_branch_raw",
    }


def test_sequence_pool_selector_cli_writes_provenance(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = sequence_pool_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--group-column",
            "candidate_branch",
            "--min-group-frame-fraction",
            "0",
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
            "4",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_sequence_pool_summary.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_sequence_pool_candidates.csv").exists()
    summary = json.loads(
        (output_dir / "mmuad_candidate_mixture_sequence_pool_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["truth_used_for_selection"] is False
    assert summary["selected_pool_by_sequence"] == {
        "seqA": "without_candidate_branch_translated",
        "seqB": "without_candidate_branch_raw",
    }


def test_sequence_pool_selector_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"][
            "raft-uav-mmuad-candidate-mixture-sequence-pool-selector"
        ]
        == "raft_uav.mmuad.candidate_mixture_map_sequence_pool_selector:main"
    )
