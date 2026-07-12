from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward_adaptive import (
    EntropyAdaptivePairBlendConfig,
    blend_candidate_posteriors as blend_entropy_only,
)
from raft_uav.mmuad.candidate_pair_forward_backward_agreement import (
    AgreementAdaptivePairBlendConfig,
    attach_agreement_adaptive_pair_prior,
    blend_candidate_posteriors,
    main as agreement_main,
)


def _candidate_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "dynamic",
                    "track_id": f"bad-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": float(10 + time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                    "predicted_sigma_m": 3.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def test_disagreement_reduces_confident_pair_weight() -> None:
    local = np.asarray([0.99, 0.01])
    pair = np.asarray([0.01, 0.99])
    _, entropy_diagnostics = blend_entropy_only(
        local,
        pair,
        config=EntropyAdaptivePairBlendConfig(
            min_pair_weight=0.0,
            max_pair_weight=1.0,
            confidence_power=1.0,
        ),
    )
    blended, diagnostics = blend_candidate_posteriors(
        local,
        pair,
        config=AgreementAdaptivePairBlendConfig(
            min_pair_weight=0.0,
            max_pair_weight=1.0,
            confidence_power=1.0,
            agreement_power=1.0,
        ),
    )

    assert diagnostics["pair_confidence"] > 0.9
    assert diagnostics["normalized_js_divergence"] > 0.9
    assert diagnostics["effective_pair_weight"] < 0.1
    assert diagnostics["effective_pair_weight"] < entropy_diagnostics["effective_pair_weight"]
    assert blended[0] > blended[1]


def test_zero_agreement_power_matches_entropy_only_blend() -> None:
    local = np.asarray([0.7, 0.2, 0.1])
    pair = np.asarray([0.1, 0.8, 0.1])
    entropy_blend, entropy_diagnostics = blend_entropy_only(
        local,
        pair,
        config=EntropyAdaptivePairBlendConfig(
            min_pair_weight=0.1,
            max_pair_weight=0.8,
            confidence_power=2.0,
        ),
    )
    agreement_blend, agreement_diagnostics = blend_candidate_posteriors(
        local,
        pair,
        config=AgreementAdaptivePairBlendConfig(
            min_pair_weight=0.1,
            max_pair_weight=0.8,
            confidence_power=2.0,
            agreement_power=0.0,
        ),
    )

    np.testing.assert_allclose(agreement_blend, entropy_blend)
    assert agreement_diagnostics["effective_pair_weight"] == pytest.approx(
        entropy_diagnostics["effective_pair_weight"]
    )


def test_attach_agreement_prior_writes_normalized_scores() -> None:
    result = attach_agreement_adaptive_pair_prior(_candidate_rows())
    rows = result.rows
    score_column = "candidate_pair_forward_backward_agreement_score"

    assert score_column in rows.columns
    assert "candidate_pair_forward_backward_agreement_normalized_js_divergence" in rows
    sums = rows.groupby(["sequence_id", "time_s"])[score_column].sum()
    np.testing.assert_allclose(sums.to_numpy(float), np.ones(len(sums)))
    assert rows["candidate_pair_forward_backward_agreement_pair_weight"].between(0, 1).all()


def test_agreement_cli_writes_candidates_and_summary(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "agreement.csv"
    summary_json = tmp_path / "agreement.json"
    _candidate_rows().to_csv(input_csv, index=False)

    status = agreement_main(
        [
            "--candidate-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--score-column",
            "ranker_score",
            "--agreement-power",
            "2",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["truth_used_for_candidate_prior"] is False
    assert payload["blend_config"]["agreement_power"] == 2.0
    assert payload["agreement_summary"]["frame_count"] == 3


def test_agreement_blend_validates_power() -> None:
    with pytest.raises(ValueError, match="agreement_power"):
        blend_candidate_posteriors(
            np.asarray([0.5, 0.5]),
            np.asarray([0.5, 0.5]),
            config=AgreementAdaptivePairBlendConfig(agreement_power=-1.0),
        )
