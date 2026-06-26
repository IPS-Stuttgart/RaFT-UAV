from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_score_calibration import (
    DEFAULT_OUTPUT_SCORE_COLUMN,
    apply_candidate_score_calibration,
    fit_candidate_score_calibration,
    load_candidate_score_calibration_model,
    save_candidate_score_calibration_model,
)


def _candidate_rows(sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for sequence_id in sequence_ids:
        records.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "candidate_branch": "raw",
                    "track_id": f"{sequence_id}-raw",
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1 if sequence_id.startswith("class0") else 0.9,
                    "confidence": 0.5,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "candidate_branch": "translated",
                    "track_id": f"{sequence_id}-translated",
                    "x_m": 10.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9 if sequence_id.startswith("class0") else 0.1,
                    "confidence": 0.5,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows(sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": list(sequence_ids),
            "time_s": [0.0] * len(sequence_ids),
            "x_m": [0.0 if sequence_id.startswith("class0") else 10.0 for sequence_id in sequence_ids],
            "y_m": [0.0] * len(sequence_ids),
            "z_m": [1.0] * len(sequence_ids),
        }
    )


def _probability_rows(sequence_ids: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for sequence_id in sequence_ids:
        class0 = 1.0 if sequence_id.startswith("class0") else 0.0
        records.append(
            {
                "sequence_id": sequence_id,
                "predicted_probability_0": class0,
                "predicted_probability_1": 1.0 - class0,
                "predicted_probability_2": 0.0,
                "predicted_probability_3": 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def test_class_conditioned_calibration_reverses_wrong_generic_branch_scores() -> None:
    train_sequences = ("class0-a", "class0-b", "class1-a", "class1-b")
    model, _, diagnostics = fit_candidate_score_calibration(
        _candidate_rows(train_sequences),
        _truth_rows(train_sequences),
        _probability_rows(train_sequences),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
        min_group_weight=1.0,
        l2_penalty=0.1,
        max_abs_logit_offset=8.0,
        include_branch_source_interactions=False,
    )

    score_sequences = ("class0-val", "class1-val")
    scored = apply_candidate_score_calibration(
        _candidate_rows(score_sequences),
        model,
        class_probabilities=_probability_rows(score_sequences),
    ).rows

    class0 = scored.loc[scored["sequence_id"] == "class0-val"].set_index("candidate_branch")
    class1 = scored.loc[scored["sequence_id"] == "class1-val"].set_index("candidate_branch")
    assert class0.loc["raw", DEFAULT_OUTPUT_SCORE_COLUMN] > class0.loc[
        "translated", DEFAULT_OUTPUT_SCORE_COLUMN
    ]
    assert class1.loc["translated", DEFAULT_OUTPUT_SCORE_COLUMN] > class1.loc[
        "raw", DEFAULT_OUTPUT_SCORE_COLUMN
    ]
    summary = diagnostics.loc[diagnostics["level"] == "summary"].iloc[0]
    assert summary["calibrated_brier"] < summary["base_brier"]


def test_candidate_score_calibration_model_round_trip_and_unseen_branch(tmp_path: Path) -> None:
    train_sequences = ("class0-a", "class0-b", "class1-a", "class1-b")
    model, _, _ = fit_candidate_score_calibration(
        _candidate_rows(train_sequences),
        _truth_rows(train_sequences),
        _probability_rows(train_sequences),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
        min_group_weight=1.0,
        l2_penalty=1.0,
        include_branch_source_interactions=False,
    )
    model_json = tmp_path / "model.json"
    save_candidate_score_calibration_model(model, model_json)
    loaded = load_candidate_score_calibration_model(model_json)

    unseen = _candidate_rows(("class0-new",)).copy()
    unseen["candidate_branch"] = "unseen"
    scored = apply_candidate_score_calibration(
        unseen,
        loaded,
        class_probabilities=_probability_rows(("class0-new",)),
    ).rows

    assert scored[DEFAULT_OUTPUT_SCORE_COLUMN].between(0.0, 1.0).all()
    assert scored["candidate_class_calibration_branch_logit_offset"].eq(0.0).all()


def test_candidate_score_calibration_entrypoints_are_exposed() -> None:
    scripts = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts["raft-uav-mmuad-fit-candidate-score-calibration"] == (
        "raft_uav.mmuad.candidate_score_calibration:fit_main"
    )
    assert scripts["raft-uav-mmuad-apply-candidate-score-calibration"] == (
        "raft_uav.mmuad.candidate_score_calibration:apply_main"
    )


def test_candidate_score_calibration_model_is_json_serializable() -> None:
    train_sequences = ("class0-a", "class0-b", "class1-a", "class1-b")
    model, _, _ = fit_candidate_score_calibration(
        _candidate_rows(train_sequences),
        _truth_rows(train_sequences),
        _probability_rows(train_sequences),
        good_threshold_m=1.0,
        max_truth_time_delta_s=0.1,
        min_group_weight=1.0,
        include_branch_source_interactions=False,
    )

    payload = json.loads(json.dumps(model))
    assert payload["schema_version"] == 1
    assert payload["output_score_column"] == DEFAULT_OUTPUT_SCORE_COLUMN
