from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.classification import infer_sequence_class_map_from_candidates
from raft_uav.mmuad.schema import CandidateFrame


def test_class_vote_defaults_missing_confidence_to_one_without_mutating_input() -> None:
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "camera",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
            },
            {
                "sequence_id": "seqA",
                "time_s": 2.0,
                "source": "camera",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "hexrotor",
            },
        ]
    )
    candidates = CandidateFrame(rows)

    class_map = infer_sequence_class_map_from_candidates(
        candidates,
        min_confidence=0.5,
    )

    assert class_map == {"seqA": "quadrotor"}
    assert "confidence" not in candidates.rows.columns
