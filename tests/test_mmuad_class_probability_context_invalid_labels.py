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
            "track_id": ["a"],
            "candidate_branch": ["static_dynamic_union"],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [1.0],
            "confidence": [0.5],
        }
    )


@pytest.mark.parametrize("bad_label", [4, -1, True, "fixed-wing"])
def test_class_probability_context_rejects_invalid_fallback_class_labels(
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
