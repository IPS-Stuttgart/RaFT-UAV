from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.classification import infer_sequence_class_map_from_candidates
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


def _candidate_frame(rows: list[dict[str, object]]) -> CandidateFrame:
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame.from_records(rows)))


def test_inferred_class_map_keeps_sequences_filtered_by_confidence() -> None:
    candidates = _candidate_frame(
        [
            {
                "sequence_id": "seq_low_confidence",
                "time_s": 0.0,
                "source": "camera",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "class_name": "quadrotor",
                "confidence": 0.1,
            },
            {
                "sequence_id": "seq_confident",
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

    class_map = infer_sequence_class_map_from_candidates(
        candidates,
        min_confidence=0.5,
        default_class="unknown",
    )

    assert class_map == {
        "seq_confident": "hexrotor",
        "seq_low_confidence": "unknown",
    }
