from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
)
from raft_uav.mmuad.candidate_pair_forward_backward_agreement_cv import (
    AgreementPairCVConfig,
    aggregate_agreement_pair_cv_folds,
    main as agreement_cv_main,
    select_agreement_pair_config_by_sequence_cv,
    write_agreement_pair_cv_outputs,
)


def _candidate_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sequence_id, offset in (("seqA", 0.0), ("seqB", 20.0)):
        for time_s in range(4):
            records.extend(
                [
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "lidar_360",
                        "track_id": f"good-{sequence_id}-{time_s}",
                        "candidate_branch": "raw",
                        "x_m": offset + float(time_s),
                        "y_m": 0.0,
                        "z_m": 1.0,
                        "candidate_reservoir_grid_score": 0.2,
                        "predicted_sigma_m": 1.0,
                    },
                    {
                        "sequence_id": sequence_id,
                        "time_s": float(time_s),
                        "source": "dynamic",
                        "track_id": f"bad-{sequence_id}-{time_s}",
                        "candidate_branch": "dynamic",
                        "x_m": offset + 10.0 + float(time_s),
                        "y_m": 0.0,
                        "z_m": 1.0,
                        "candidate_reservoir_grid_score": 0.8,
                        "predicted_sigma_m": 8.0,
                    },
                ]
            )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    records = []
    for sequence_id, offset in (("seqA", 0.0), ("seqB", 20.0)):
        for time_s in range(4):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(time_s),
                    "x_m": offset + float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _single_grid_selection() -> tuple[dict, pd.DataFrame, pd.DataFrame, object]:
    return select_agreement_pair_config_by_sequence_cv(
        _candidate_rows(),
        _truth_rows(),
        pair_config=CandidatePairForwardBackwardConfig(
            transition_distance_std_m=5.0,
            transition_speed_std_mps=20.0,
            acceleration_std_mps2=20.0,
        ),
        mixture_config=CandidateMixtureMapConfig(
            top_k=2,
            smoothness_weight=10.0,
            iterations=2,
        ),
        cv_config=AgreementPairCVConfig(
            selection_metric="mse_3d_m",
            risk_aversion=0.5,
            tail_quantile=1.0,
        ),
        min_pair_weights=(0.0,),
        max_pair_weights=(1.0,),
        entropy_powers=(1.0,),
        agreement_powers=(1.0,),
        agreement_floors=(0.0,),
    )


def test_risk_aware_aggregate_can_prefer_lower_tail_error() -> None:
    rows = pd.DataFrame(
        [
            {
                "grid_label": "mean-best",
                "holdout_sequence_id": sequence,
                "min_pair_weight": 0.0,
                "max_pair_weight": 1.0,
                "entropy_power": 1.0,
                "agreement_power": 1.0,
                "agreement_floor": 0.0,
                "mse_3d_m": value,
            }
            for sequence, value in zip(
                ("a", "b", "c"),
                (1.0, 1.0, 7.0),
                strict=True,
            )
        ]
        + [
            {
                "grid_label": "tail-safe",
                "holdout_sequence_id": sequence,
                "min_pair_weight": 0.1,
                "max_pair_weight": 0.75,
                "entropy_power": 2.0,
                "agreement_power": 1.0,
                "agreement_floor": 0.1,
                "mse_3d_m": 3.1,
            }
            for sequence in ("a", "b", "c")
        ]
    )

    mean_only = aggregate_agreement_pair_cv_folds(
        rows,
        cv_config=AgreementPairCVConfig(
            selection_metric="mse_3d_m",
            risk_aversion=0.0,
            tail_quantile=1.0,
        ),
        expected_sequence_count=3,
    )
    risk_aware = aggregate_agreement_pair_cv_folds(
        rows,
        cv_config=AgreementPairCVConfig(
            selection_metric="mse_3d_m",
            risk_aversion=0.75,
            tail_quantile=1.0,
        ),
        expected_sequence_count=3,
    )

    assert mean_only.iloc[0]["grid_label"] == "mean-best"
    assert risk_aware.iloc[0]["grid_label"] == "tail-safe"


def test_selection_writes_frozen_config_and_normalized_candidates(
    tmp_path: Path,
) -> None:
    selected, folds, aggregate, candidates = _single_grid_selection()
    paths = write_agreement_pair_cv_outputs(
        selected_config=selected,
        fold_summary=folds,
        aggregate_summary=aggregate,
        selected_candidates=candidates,
        output_dir=tmp_path,
    )

    assert len(folds) == 2
    assert len(aggregate) == 1
    assert selected["selection_protocol"] == "train-sequence-cv-aggregate"
    assert paths["selected_config_json"].exists()
    payload = json.loads(paths["selected_config_json"].read_text(encoding="utf-8"))
    assert payload["truth_used_for_candidate_prior"] is False
    sums = candidates.rows.groupby(["sequence_id", "time_s"])[
        "candidate_pair_forward_backward_agreement_adaptive_score"
    ].sum()
    assert np.allclose(sums.to_numpy(float), 1.0)


def test_cli_writes_selection_artifacts(tmp_path: Path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "selection"
    _candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = agreement_cv_main(
        [
            "--candidate-csv",
            str(candidate_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--min-pair-weight",
            "0",
            "--max-pair-weight",
            "1",
            "--entropy-power",
            "1",
            "--agreement-power",
            "1",
            "--agreement-floor",
            "0",
            "--mixture-top-k",
            "2",
            "--mixture-smoothness-weight",
            "10",
            "--mixture-iterations",
            "2",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_agreement_pair_cv_folds.csv").exists()
    assert (output_dir / "mmuad_agreement_pair_cv_aggregate.csv").exists()
    assert (output_dir / "mmuad_agreement_pair_cv_selected_config.json").exists()
    assert (output_dir / "mmuad_agreement_pair_cv_selected_candidates.csv").exists()


def test_cv_rejects_single_sequence() -> None:
    with pytest.raises(ValueError, match="at least two"):
        select_agreement_pair_config_by_sequence_cv(
            _candidate_rows().loc[lambda frame: frame["sequence_id"] == "seqA"],
            _truth_rows().loc[lambda frame: frame["sequence_id"] == "seqA"],
            min_pair_weights=(0.0,),
            max_pair_weights=(1.0,),
            entropy_powers=(1.0,),
            agreement_powers=(1.0,),
            agreement_floors=(0.0,),
        )
