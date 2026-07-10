from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_map_multistart import CandidateMixtureMultiStartConfig
from raft_uav.mmuad.candidate_pair_log_multistart import (
    DEFAULT_OUTPUT_SCORE_COLUMN,
    PairLogPosteriorConfig,
    attach_pair_log_posterior_score,
    main as pair_log_main,
    run_pair_log_multistart,
)


def _single_frame_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "source": "lidar",
                "track_id": "preferred",
                "candidate_branch": "raw",
                "x_m": -1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "candidate_pair_forward_backward_score": 0.9,
                "predicted_sigma_m": 10.0,
            },
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "other",
                "candidate_branch": "translated",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "candidate_pair_forward_backward_score": 0.1,
                "predicted_sigma_m": 10.0,
            },
        ]
    )


def _trajectory_candidates() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "001",
                    "time_s": float(time_s),
                    "source": "lidar",
                    "track_id": f"good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "candidate_pair_forward_backward_score": 0.9,
                    "candidate_pair_forward_backward_log_probability": float(np.log(0.9)),
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "001",
                    "time_s": float(time_s),
                    "source": "radar",
                    "track_id": f"bad-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "candidate_pair_forward_backward_score": 0.1,
                    "candidate_pair_forward_backward_log_probability": float(np.log(0.1)),
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["001", "001", "001"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def _mixture_config(score_column: str) -> core.CandidateMixtureMapConfig:
    return core.CandidateMixtureMapConfig(
        top_k=0,
        score_column=score_column,
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
        loss="huber",
        huber_delta=1.0,
        smoothness_weight=10.0,
        iterations=3,
    )


def test_pair_log_score_normalizes_and_preserves_opaque_sequence_id() -> None:
    adapted = attach_pair_log_posterior_score(_single_frame_candidates())

    assert adapted["sequence_id"].tolist() == ["001", "001"]
    probability = adapted["candidate_pair_mixture_probability"].to_numpy(float)
    assert probability.sum() == pytest.approx(1.0)
    assert probability.tolist() == pytest.approx([0.9, 0.1])
    score = adapted[DEFAULT_OUTPUT_SCORE_COLUMN].to_numpy(float)
    assert score[0] - score[1] == pytest.approx(np.log(9.0))
    assert set(adapted["candidate_pair_mixture_score_source"]) == {
        "probability-fallback"
    }


def test_log_posterior_has_the_expected_mixture_strength() -> None:
    candidates = _single_frame_candidates()
    probability_responsibility = core.compute_candidate_responsibilities(
        candidates,
        np.asarray([0.0, 0.0, 0.0]),
        config=_mixture_config("candidate_pair_forward_backward_score"),
    )
    adapted = attach_pair_log_posterior_score(candidates)
    log_responsibility = core.compute_candidate_responsibilities(
        adapted,
        np.asarray([0.0, 0.0, 0.0]),
        config=_mixture_config(DEFAULT_OUTPUT_SCORE_COLUMN),
    )

    probability_weight = float(
        probability_responsibility.loc[
            probability_responsibility["track_id"] == "preferred",
            "mixture_responsibility",
        ].iloc[0]
    )
    log_weight = float(
        log_responsibility.loc[
            log_responsibility["track_id"] == "preferred",
            "mixture_responsibility",
        ].iloc[0]
    )
    assert probability_weight == pytest.approx(1.0 / (1.0 + np.exp(-0.8)))
    assert log_weight == pytest.approx(0.9)
    assert log_weight > probability_weight + 0.20


def test_pair_log_multistart_uses_log_score_without_truth_selection() -> None:
    result = run_pair_log_multistart(
        _trajectory_candidates(),
        mixture_config=_mixture_config(DEFAULT_OUTPUT_SCORE_COLUMN),
        multistart_config=CandidateMixtureMultiStartConfig(
            include_score_top1=True,
            include_frame_median=False,
            include_branch_starts=True,
        ),
        truth=_truth(),
    )

    assert result.summary["truth_used_for_selection"] is False
    assert result.summary["mixture_score_space"] == "log-posterior"
    assert result.summary["mixture_config"]["score_column"] == DEFAULT_OUTPUT_SCORE_COLUMN
    assert result.summary["pair_log_posterior"]["posterior_sum_error_max"] < 1.0e-12
    assert result.multistart.selected_result.summary["metrics"]["pooled"]["rmse_3d_m"] < 0.1


def test_pair_log_multistart_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "pair_candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _trajectory_candidates().to_csv(candidates_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = pair_log_main(
        [
            "--pair-candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--smoothness-weight",
            "10",
            "--iterations",
            "2",
            "--no-frame-median-start",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_pair_log_multistart_candidates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_multistart_summary.csv").exists()
    summary_path = output_dir / "mmuad_pair_log_multistart_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["mixture_score_space"] == "log-posterior"
    assert payload["pair_log_posterior"]["posterior_sum_error_max"] < 1.0e-12
