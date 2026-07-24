from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.classification import infer_sequence_class_map_from_candidates
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


def _candidate_frame(rows: list[dict[str, object]]) -> CandidateFrame:
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(rows)))


@pytest.mark.parametrize("bad_confidence", [np.inf, -np.inf])
def test_inferred_class_map_rejects_nonfinite_candidate_confidences(
    bad_confidence: float,
) -> None:
    candidates = _candidate_frame(
        [
            {
                "sequence_id": "seq",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
                "confidence": 0.9,
            },
            {
                "sequence_id": "seq",
                "time_s": 1.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "malformed",
                "confidence": bad_confidence,
            },
        ]
    )

    with pytest.raises(ValueError, match="candidate confidence"):
        infer_sequence_class_map_from_candidates(candidates)


def test_inferred_class_map_preserves_missing_confidence_default() -> None:
    candidates = _candidate_frame(
        [
            {
                "sequence_id": "seq",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
            }
        ]
    )

    assert infer_sequence_class_map_from_candidates(candidates) == {
        "seq": "quadrotor"
    }
