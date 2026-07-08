from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import (
    attach_class_probability_context,
    main as class_probability_context_main,
)
from raft_uav.mmuad.class_probability_csv import read_class_probability_csv
from raft_uav.mmuad.schema import CandidateFrame


def test_class_probability_csv_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "sequence_id,predicted_probability_0,predicted_probability_1,"
        "predicted_probability_2,predicted_probability_3\n"
        "001,0.1,0.2,0.6,0.1\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["a"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    probabilities = read_class_probability_csv(probabilities_csv)
    augmented = attach_class_probability_context(
        CandidateFrame(candidates),
        probabilities,
        interaction_columns=(),
        fill_missing="error",
    ).rows

    assert probabilities.loc[0, "sequence_id"] == "001"
    assert augmented.loc[0, "image_class_probability_available"] == pytest.approx(1.0)
    assert augmented.loc[0, "image_class_prob_2"] == pytest.approx(0.6)


def test_class_probability_csv_canonicalizes_shared_sequence_alias_ids(
    tmp_path: Path,
) -> None:
    probabilities_csv = tmp_path / "probabilities_clip.csv"
    probabilities_csv.write_text(
        "clip_id,predicted_probability_0,predicted_probability_1,"
        "predicted_probability_2,predicted_probability_3\n"
        "001,0.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["a"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    probabilities = read_class_probability_csv(probabilities_csv)
    augmented = attach_class_probability_context(
        CandidateFrame(candidates),
        probabilities,
        interaction_columns=(),
        fill_missing="error",
    ).rows

    assert probabilities.loc[0, "sequence_id"] == "001"
    assert probabilities.loc[0, "clip_id"] == "001"
    assert augmented.loc[0, "image_class_probability_available"] == pytest.approx(1.0)
    assert augmented.loc[0, "image_class_prob_1"] == pytest.approx(1.0)


def test_class_probability_context_cli_preserves_zero_padded_probability_ids(
    tmp_path: Path,
) -> None:
    candidate_json = tmp_path / "candidates.json"
    candidate_json.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "sequence_id": "001",
                        "time_s": 0.0,
                        "source": "lidar_360",
                        "track_id": "candidate-a",
                        "x_m": 1.0,
                        "y_m": 0.0,
                        "z_m": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        "sequence_id,predicted_probability_0,predicted_probability_1,"
        "predicted_probability_2,predicted_probability_3\n"
        "001,0.1,0.2,0.6,0.1\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "augmented.csv"

    status = class_probability_context_main(
        [
            "--candidate-csv",
            str(candidate_json),
            "--class-probabilities-csv",
            str(probabilities_csv),
            "--output-csv",
            str(output_csv),
            "--interaction-column",
            "confidence",
            "--fill-missing",
            "error",
        ]
    )

    assert status == 0
    augmented = pd.read_csv(output_csv, dtype={"sequence_id": str})
    assert augmented.loc[0, "sequence_id"] == "001"
    assert augmented.loc[0, "image_class_probability_available"] == pytest.approx(1.0)
    assert augmented.loc[0, "image_class_prob_2"] == pytest.approx(0.6)
