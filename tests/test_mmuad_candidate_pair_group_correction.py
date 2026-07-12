from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_group_correction import (
    PairGroupMultiplicityConfig,
    attach_group_corrected_pair_prior,
    main as group_correction_main,
    prepare_group_corrected_pair_candidates,
)


def _duplicate_group_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["raw", "translated", "dynamic"],
            "candidate_branch": ["raw", "translated", "dynamic"],
            "track_id": ["duplicate-a", "duplicate-b", "singleton"],
            "candidate_origin_row": ["physical-a", "physical-a", "physical-b"],
            "x_m": [0.0, 0.1, 10.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "ranker_score": [0.0, 0.0, 0.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
            "confidence": [1.0, 1.0, 1.0],
        }
    )


def _pair_config() -> CandidatePairForwardBackwardConfig:
    return CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        fallback_score_columns=(),
        sigma_column="predicted_sigma_m",
        score_normalization="none",
        score_weight=1.0,
        sigma_log_weight=0.0,
        output_score_column="test_pair_score",
    )


def test_group_correction_removes_duplicate_emission_mass_advantage() -> None:
    augmented, _, summary = attach_group_corrected_pair_prior(
        _duplicate_group_candidates(),
        pair_config=_pair_config(),
        group_config=PairGroupMultiplicityConfig(correction_strength=1.0),
    )
    rows = augmented.rows
    duplicate_mass = float(
        rows.loc[
            rows["candidate_origin_row"] == "physical-a",
            "test_pair_score",
        ].sum()
    )
    singleton_mass = float(
        rows.loc[
            rows["candidate_origin_row"] == "physical-b",
            "test_pair_score",
        ].sum()
    )

    assert duplicate_mass == pytest.approx(0.5)
    assert singleton_mass == pytest.approx(0.5)
    assert float(rows["test_pair_score"].sum()) == pytest.approx(1.0)
    assert summary["duplicate_candidate_rows"] == 2


def test_zero_correction_recovers_flat_candidate_posterior() -> None:
    augmented, _, _ = attach_group_corrected_pair_prior(
        _duplicate_group_candidates(),
        pair_config=_pair_config(),
        group_config=PairGroupMultiplicityConfig(correction_strength=0.0),
    )
    rows = augmented.rows.sort_values("track_id")

    assert rows["test_pair_score"].to_numpy(float) == pytest.approx(
        np.full(3, 1.0 / 3.0)
    )


def test_prepare_group_correction_preserves_coordinates_and_branches() -> None:
    original = _duplicate_group_candidates()
    prepared, effective, summary = prepare_group_corrected_pair_candidates(
        original,
        pair_config=_pair_config(),
        group_config=PairGroupMultiplicityConfig(correction_strength=0.75),
    )

    assert prepared[["x_m", "y_m", "z_m"]].equals(
        original[["x_m", "y_m", "z_m"]]
    )
    assert prepared["candidate_branch"].tolist() == original["candidate_branch"].tolist()
    assert effective.score_normalization == "none"
    assert effective.score_column == "candidate_pair_group_corrected_emission_score"
    assert summary["resolved_group_column"] == "candidate_origin_row"


def test_missing_group_policy_error_rejects_unidentified_candidates() -> None:
    candidates = _duplicate_group_candidates().drop(columns=["candidate_origin_row"])

    with pytest.raises(ValueError, match="no hypothesis-group column found"):
        prepare_group_corrected_pair_candidates(
            candidates,
            pair_config=_pair_config(),
            group_config=PairGroupMultiplicityConfig(missing_group_policy="error"),
        )


def test_textual_missing_group_ids_remain_unique() -> None:
    candidates = _duplicate_group_candidates()
    candidates["candidate_origin_row"] = ["none", "none", "<NA>"]

    prepared, _, summary = prepare_group_corrected_pair_candidates(
        candidates,
        pair_config=_pair_config(),
        group_config=PairGroupMultiplicityConfig(missing_group_policy="unique"),
    )

    assert prepared["candidate_pair_hypothesis_group_size"].tolist() == [1, 1, 1]
    assert prepared["candidate_pair_hypothesis_group"].nunique() == 3
    assert summary["duplicate_candidate_rows"] == 0


@pytest.mark.parametrize("missing_group_id", ["none", "NONE", "NaN", "<NA>"])
def test_missing_group_policy_error_rejects_textual_missing_ids(
    missing_group_id: str,
) -> None:
    candidates = _duplicate_group_candidates().iloc[[0]].copy()
    candidates["candidate_origin_row"] = missing_group_id

    with pytest.raises(ValueError, match="has missing values at rows"):
        prepare_group_corrected_pair_candidates(
            candidates,
            pair_config=_pair_config(),
            group_config=PairGroupMultiplicityConfig(missing_group_policy="error"),
        )


def test_group_correction_supports_agreement_adaptive_output() -> None:
    from raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive import (
        AgreementAdaptivePairBlendConfig,
    )

    augmented, _, _ = attach_group_corrected_pair_prior(
        _duplicate_group_candidates(),
        pair_config=_pair_config(),
        group_config=PairGroupMultiplicityConfig(correction_strength=1.0),
        blend_config=AgreementAdaptivePairBlendConfig(
            min_pair_weight=1.0,
            max_pair_weight=1.0,
            output_score_column="adaptive_score",
        ),
    )

    assert float(augmented.rows["adaptive_score"].sum()) == pytest.approx(1.0)
    assert np.isfinite(augmented.rows["adaptive_score"]).all()


def test_group_correction_cli_writes_provenance(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "corrected.csv"
    summary_json = tmp_path / "summary.json"
    _duplicate_group_candidates().to_csv(candidate_csv, index=False)

    rc = group_correction_main(
        [
            "--candidate-csv",
            str(candidate_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--score-column",
            "ranker_score",
            "--score-normalization",
            "none",
            "--sigma-log-weight",
            "0",
            "--pair-score-column",
            "pair_score",
            "--correction-strength",
            "1",
        ]
    )

    assert rc == 0
    written = pd.read_csv(output_csv)
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert written["pair_score"].sum() == pytest.approx(1.0)
    assert payload["truth_used_for_candidate_prior"] is False
    assert payload["grouping_summary"]["resolved_group_column"] == "candidate_origin_row"
    assert payload["group_config"]["correction_strength"] == 1.0
