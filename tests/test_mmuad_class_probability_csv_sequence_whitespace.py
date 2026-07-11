from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.class_probability_csv import read_class_probability_csv
from raft_uav.mmuad.schema import CandidateFrame


def test_class_probability_csv_strips_sequence_id_values_before_join(
    tmp_path: Path,
) -> None:
    probabilities_csv = tmp_path / "probabilities.csv"
    probabilities_csv.write_text(
        " Sequence ,predicted_probability_0,predicted_probability_1,"
        "predicted_probability_2,predicted_probability_3\n"
        " 001 ,0.1,0.2,0.6,0.1\n",
        encoding="utf-8",
    )
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["001"],
                "time_s": [0.0],
                "source": ["lidar_360"],
                "track_id": ["candidate-a"],
                "x_m": [1.0],
                "y_m": [0.0],
                "z_m": [0.0],
            }
        )
    )

    probabilities = read_class_probability_csv(probabilities_csv)
    augmented = attach_class_probability_context(
        candidates,
        probabilities,
        interaction_columns=(),
        fill_missing="error",
    ).rows

    assert probabilities.loc[0, "sequence_id"] == "001"
    assert augmented.loc[0, "image_class_probability_available"] == pytest.approx(1.0)
    assert augmented.loc[0, "image_class_prob_2"] == pytest.approx(0.6)
