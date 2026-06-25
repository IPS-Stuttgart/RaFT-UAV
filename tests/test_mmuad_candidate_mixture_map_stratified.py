from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
)
from raft_uav.mmuad.candidate_mixture_map_stratified import (
    StratifiedMixtureTopKConfig,
    main as stratified_main,
    run_stratified_candidate_mixture_map,
    select_stratified_mixture_candidates,
)


def _crowded_candidates() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_s in range(5):
        records.append(
            {
                "sequence_id": "seqA",
                "time_s": float(time_s),
                "source": "lidar_360",
                "track_id": f"raw-good-{time_s}",
                "candidate_branch": "raw",
                "x_m": float(time_s),
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.10,
                "predicted_sigma_m": 1.0,
            }
        )
        for index, score in enumerate((0.99, 0.98, 0.97)):
            records.append(
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"translated-bad-{time_s}-{index}",
                    "candidate_branch": "source_translation",
                    "x_m": float(time_s + 20 + index),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": score,
                    "predicted_sigma_m": 20.0,
                }
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


def _mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=2,
        score_column="ranker_score",
        sigma_column="predicted_sigma_m",
        smoothness_weight=100.0,
        iterations=5,
        loss="huber",
        huber_delta=1.0,
    )


def test_stratified_topk_preserves_low_score_raw_branch() -> None:
    selected = select_stratified_mixture_candidates(
        _crowded_candidates(),
        config=StratifiedMixtureTopKConfig(
            top_k=2,
            min_per_branch=1,
            min_per_source=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
        ),
    )

    for _, frame in selected.groupby(["sequence_id", "time_s"]):
        assert len(frame) == 2
        assert set(frame["candidate_branch"]) == {"raw", "source_translation"}
        raw_reason = frame.loc[
            frame["candidate_branch"] == "raw", "mixture_stratified_reason"
        ].iloc[0]
        assert "branch:raw" in raw_reason


def test_stratified_topk_improves_mixture_when_global_topk_buries_good_branch() -> None:
    candidates = _crowded_candidates()
    truth = _truth_rows()
    naive = run_candidate_mixture_map(
        candidates,
        config=_mixture_config(),
        truth=truth,
    )
    stratified = run_stratified_candidate_mixture_map(
        candidates,
        stratified_config=StratifiedMixtureTopKConfig(
            top_k=2,
            min_per_branch=1,
            min_per_source=0,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
        ),
        mixture_config=_mixture_config(),
        truth=truth,
    )

    naive_rmse = float(naive.summary["metrics"]["pooled"]["rmse_3d_m"])
    stratified_rmse = float(
        stratified.mixture_result.summary["metrics"]["pooled"]["rmse_3d_m"]
    )
    assert naive_rmse > 10.0
    assert stratified_rmse < 0.1
    assert stratified_rmse < naive_rmse
    assert stratified.selection_summary["frames_all_branches_preserved_fraction"] == 1.0


def test_stratified_mixture_cli_writes_selection_and_mixture_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _crowded_candidates().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = stratified_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--min-per-branch",
            "1",
            "--min-per-source",
            "0",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "100",
            "--iterations",
            "5",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_stratified_mixture_candidates.csv").exists()
    assert (output_dir / "mmuad_stratified_mixture_selection_summary.json").exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_assignments.csv").exists()
    summary = json.loads(
        (output_dir / "mmuad_stratified_mixture_selection_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["frames_all_branches_preserved_fraction"] == 1.0


def test_stratified_candidate_mixture_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-stratified-candidate-mixture-map"]
        == "raft_uav.mmuad.candidate_mixture_map_stratified:main"
    )
