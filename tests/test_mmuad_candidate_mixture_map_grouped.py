from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    compute_candidate_responsibilities,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    HypothesisGroupConfig,
    compute_grouped_candidate_responsibilities,
    main as grouped_main,
    prepare_hypothesis_group_candidates,
)


def _duplicate_hypotheses() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "cluster-1@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "cluster-1@calibrated",
                "candidate_branch": "source_translation_calibrated",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "cluster-2@raw",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 2,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _neutral_mixture_config() -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(
        top_k=0,
        score_column="ranker_score",
        score_normalization="none",
        score_weight=0.0,
        sigma_log_weight=0.0,
        loss="squared",
        smoothness_weight=0.0,
        iterations=1,
    )


def test_group_correction_removes_duplicate_branch_mass_advantage() -> None:
    rows = _duplicate_hypotheses()
    baseline = compute_candidate_responsibilities(
        rows,
        np.zeros(3),
        config=_neutral_mixture_config(),
    )
    baseline_group_mass = baseline.groupby(
        "mmuad_calibration_origin_row"
    )["mixture_responsibility"].sum()
    assert baseline_group_mass.loc[1] == pytest.approx(2.0 / 3.0)
    assert baseline_group_mass.loc[2] == pytest.approx(1.0 / 3.0)

    grouped = compute_grouped_candidate_responsibilities(
        rows,
        np.zeros(3),
        mixture_config=_neutral_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
    )
    group_mass = grouped.groupby(
        "mixture_hypothesis_group"
    )["mixture_responsibility"].sum()
    assert sorted(group_mass.to_numpy(float)) == pytest.approx([0.5, 0.5])
    sibling_weights = grouped.loc[
        grouped["mmuad_calibration_origin_row"] == 1,
        "mixture_responsibility",
    ]
    assert sibling_weights.tolist() == pytest.approx([0.25, 0.25])


def test_missing_group_column_uses_unique_rows_and_preserves_baseline() -> None:
    rows = _duplicate_hypotheses().drop(columns=["mmuad_calibration_origin_row"])
    baseline = compute_candidate_responsibilities(
        rows,
        np.zeros(3),
        config=_neutral_mixture_config(),
    )
    grouped = compute_grouped_candidate_responsibilities(
        rows,
        np.zeros(3),
        mixture_config=_neutral_mixture_config(),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
    )

    assert grouped["mixture_hypothesis_group_size"].tolist() == [1, 1, 1]
    assert grouped["mixture_responsibility"].to_numpy(float) == pytest.approx(
        baseline["mixture_responsibility"].to_numpy(float)
    )


def test_prepare_group_candidates_rejects_missing_required_group() -> None:
    rows = _duplicate_hypotheses().drop(columns=["mmuad_calibration_origin_row"])
    with pytest.raises(ValueError, match="no hypothesis group column"):
        prepare_hypothesis_group_candidates(
            rows,
            mixture_config=_neutral_mixture_config(),
            group_config=HypothesisGroupConfig(missing_group_policy="error"),
        )


def test_grouped_mixture_cli_writes_group_diagnostics(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output_dir = tmp_path / "out"
    _duplicate_hypotheses().to_csv(candidates, index=False)

    status = grouped_main(
        [
            "--candidates-csv",
            str(candidates),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "0",
            "--score-column",
            "ranker_score",
            "--score-normalization",
            "none",
            "--score-weight",
            "0",
            "--sigma-log-weight",
            "0",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    corrected = pd.read_csv(output_dir / "mmuad_group_corrected_candidates.csv")
    assert "mixture_hypothesis_group_size" in corrected.columns
    assignments = pd.read_csv(output_dir / "mmuad_candidate_mixture_assignments.csv")
    assert "mixture_hypothesis_group_mass" in assignments.columns
    summary = json.loads(
        (output_dir / "mmuad_hypothesis_group_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["duplicate_hypothesis_group_count"] == 1


def test_grouped_candidate_mixture_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"][
            "raft-uav-mmuad-grouped-candidate-mixture-map"
        ]
        == "raft_uav.mmuad.candidate_mixture_map_grouped:main"
    )
