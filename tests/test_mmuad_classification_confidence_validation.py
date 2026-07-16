from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.classification import infer_sequence_class_map_from_candidates
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


def _candidates() -> CandidateFrame:
    rows = pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seq_low",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
                "confidence": 0.1,
            },
            {
                "sequence_id": "seq_high",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "hexrotor",
                "confidence": 0.9,
            },
        ]
    )
    return CandidateFrame(normalize_candidate_columns(rows))


@pytest.mark.parametrize(
    "bad_threshold",
    [
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        pd.NA,
        np.array([0.5]),
    ],
)
def test_inferred_class_map_rejects_invalid_confidence_thresholds(
    bad_threshold: object,
) -> None:
    with pytest.raises(ValueError, match="min_confidence must be a finite real scalar"):
        infer_sequence_class_map_from_candidates(
            _candidates(),
            min_confidence=bad_threshold,
        )


@pytest.mark.parametrize("threshold", [0.5, np.float64(0.5), np.array(0.5), "0.5"])
def test_inferred_class_map_accepts_finite_scalar_confidence_thresholds(
    threshold: object,
) -> None:
    class_map = infer_sequence_class_map_from_candidates(
        _candidates(),
        min_confidence=threshold,
        default_class="unknown",
    )

    assert class_map == {
        "seq_high": "hexrotor",
        "seq_low": "unknown",
    }
