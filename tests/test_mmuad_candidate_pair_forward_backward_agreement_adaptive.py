from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive import (
    AgreementAdaptivePairBlendConfig,
    agreement_adaptive_pair_summary,
    attach_agreement_adaptive_pair_prior,
    blend_candidate_posteriors,
    main as agreement_main,
)


def test_confident_but_contradictory_pair_posterior_backs_off() -> None:
    blended, diagnostics = blend_candidate_posteriors(
        np.asarray([0.999, 0.001]),
        np.asarray([0.001, 0.999]),
        config=AgreementAdaptivePairBlendConfig(
            entropy_power=1.0,
            agreement_power=2.0,
            agreement_floor=0.0,
        ),
    )

    assert diagnostics["pair_confidence"] > 0.9
    assert diagnostics["local_pair_js_divergence"] > 0.9
    assert diagnostics["local_pair_agreement"] < 0.1
    assert diagnostics["effective_pair_weight"] < 0.05
    assert blended[0] > blended[1]


def test_confident_agreeing_pair_posterior_keeps_temporal_influence() -> None:
    blended, diagnostics = blend_candidate_posteriors(
        np.asarray([0.9, 0.1]),
        np.asarray([0.999, 0.001]),
        config=AgreementAdaptivePairBlendConfig(
            entropy_power=1.0,
            agreement_power=1.0,
            agreement_floor=0.0,
        ),
    )

    assert diagnostics["pair_confidence"] > 0.9
    assert diagnostics["local_pair_agreement"] > 0.7
    assert diagnostics["effective_pair_weight"] > 0.7
    assert blended[0] > 0.95
    assert float(np.sum(blended)) == pytest.approx(1.0)


def _trajectory_candidates() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "good-0",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "good-1",
                "candidate_branch": "raw",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.4,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "dynamic",
                "track_id": "bad-1",
                "candidate_branch": "dynamic",
                "x_m": 20.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.6,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 2.0,
                "source": "lidar_360",
                "track_id": "good-2",
                "candidate_branch": "raw",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def _pair_config() -> CandidatePairForwardBackwardConfig:
    return CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        sigma_column="predicted_sigma_m",
        transition_distance_std_m=1.0,
        transition_speed_std_mps=0.0,
        max_speed_mps=5.0,
        speed_gate_penalty=100.0,
        acceleration_std_mps2=2.0,
        max_acceleration_mps2=10.0,
        acceleration_gate_penalty=100.0,
        source_switch_penalty=0.0,
        branch_switch_penalty=0.0,
        track_continuation_bonus=0.0,
    )


def test_agreement_adaptive_prior_writes_normalized_diagnostics() -> None:
    augmented = attach_agreement_adaptive_pair_prior(
        _trajectory_candidates(),
        pair_config=_pair_config(),
        blend_config=AgreementAdaptivePairBlendConfig(agreement_power=2.0),
    ).rows

    score_column = "candidate_pair_forward_backward_agreement_adaptive_score"
    frame_sums = augmented.groupby(["sequence_id", "time_s"])[score_column].sum()
    assert frame_sums.to_numpy() == pytest.approx(np.ones(len(frame_sums)))
    assert {
        "candidate_pair_forward_backward_agreement_local_pair_js_divergence",
        "candidate_pair_forward_backward_agreement_local_pair_agreement",
        "candidate_pair_forward_backward_agreement_effective_pair_weight",
    }.issubset(augmented.columns)

    summary = agreement_adaptive_pair_summary(augmented)
    assert summary["frame_count"] == 3
    assert summary["posterior_sum_error_max"] < 1.0e-12


def test_agreement_adaptive_cli_writes_candidates_and_summary(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "agreement.csv"
    summary_json = tmp_path / "agreement.json"
    _trajectory_candidates().to_csv(candidate_csv, index=False)

    status = agreement_main(
        [
            "--candidate-csv",
            str(candidate_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--score-column",
            "ranker_score",
            "--pair-score-column",
            "custom_pair_score",
            "--min-pair-weight",
            "0",
            "--max-pair-weight",
            "0",
            "--transition-distance-std-m",
            "1",
            "--transition-speed-std-mps",
            "0",
            "--max-speed-mps",
            "5",
            "--speed-gate-penalty",
            "100",
            "--acceleration-std-mps2",
            "2",
            "--max-acceleration-mps2",
            "10",
            "--acceleration-gate-penalty",
            "100",
            "--agreement-power",
            "2",
        ]
    )

    assert status == 0
    written = pd.read_csv(output_csv)
    assert "candidate_pair_forward_backward_agreement_adaptive_score" in written.columns
    assert "custom_pair_score" in written.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["truth_used_for_candidate_prior"] is False
    assert payload["agreement_adaptive_summary"]["frame_count"] == 3
    assert payload["agreement_adaptive_summary"]["pair_score_column"] == (
        "custom_pair_score"
    )
    assert payload["agreement_adaptive_summary"][
        "adaptive_top_differs_from_pair_fraction"
    ] == pytest.approx(1.0 / 3.0)
    assert payload["blend_config"]["agreement_power"] == 2.0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("agreement_power", 0.0, "agreement_power"),
        ("agreement_floor", -0.1, "agreement_floor"),
        ("agreement_floor", 1.1, "agreement_floor"),
    ],
)
def test_agreement_adaptive_blend_rejects_invalid_controls(
    field: str,
    value: float,
    match: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        blend_candidate_posteriors(
            np.asarray([0.5, 0.5]),
            np.asarray([0.5, 0.5]),
            config=AgreementAdaptivePairBlendConfig(**kwargs),
        )
