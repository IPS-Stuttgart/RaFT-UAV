from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.class_probability_context import attach_class_probability_context
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["lidar_360"],
            "track_id": ["candidate"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
            "confidence": [0.5],
        }
    )


@pytest.mark.parametrize("bad_label", [1.000001, 2.999999, "1.000001"])
def test_class_probability_context_rejects_near_integer_class_labels(
    bad_label: object,
) -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "predicted_class": [bad_label],
        }
    )

    with pytest.raises(ValueError, match="official Track 5 class IDs"):
        attach_class_probability_context(
            CandidateFrame(_candidate_rows()),
            probabilities,
            interaction_columns=(),
        )


@pytest.mark.parametrize("class_label", [1, 1.0, "1.0"])
def test_class_probability_context_accepts_exact_integer_equivalents(
    class_label: object,
) -> None:
    probabilities = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "predicted_class": [class_label],
        }
    )

    augmented = attach_class_probability_context(
        CandidateFrame(_candidate_rows()),
        probabilities,
        interaction_columns=(),
    ).rows

    assert augmented.loc[0, "image_class_prob_1"] == pytest.approx(1.0)
    assert augmented.loc[0, "image_predicted_class_id"] == pytest.approx(1.0)
