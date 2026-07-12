from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_branch_context import attach_candidate_branch_context
from raft_uav.mmuad.schema import CandidateFrame


def test_branch_context_reuses_generator_interactions_for_every_branch() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq", "seq"],
                "time_s": [0.0, 0.0],
                "source": ["lidar", "radar"],
                "track_id": ["raw", "translated"],
                "candidate_branch": ["raw", "translated"],
                "x_m": [0.0, 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [0.0, 0.0],
                "confidence": [0.25, 0.75],
            }
        )
    )

    augmented = attach_candidate_branch_context(
        candidates,
        interaction_columns=(column for column in ("confidence",)),
    ).rows

    assert "image_candidate_branch_raw_x_confidence" in augmented.columns
    assert "image_candidate_branch_translated_x_confidence" in augmented.columns
    assert augmented["image_candidate_branch_raw_x_confidence"].tolist() == [0.25, 0.0]
    assert augmented["image_candidate_branch_translated_x_confidence"].tolist() == [0.0, 0.75]
