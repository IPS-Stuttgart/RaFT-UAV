from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map_sequence_pool_selector import (
    CandidatePoolSequenceSelectorConfig,
    build_sequence_candidate_pool_variants,
)


def _candidate_rows(branches: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for time_s in (0.0, 1.0):
        for branch in branches:
            records.append(
                {
                    "sequence_id": "seq-a",
                    "time_s": time_s,
                    "candidate_branch": branch,
                    "source": branch,
                    "track_id": f"{branch}-{time_s}",
                    "x_m": time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                }
            )
    return pd.DataFrame.from_records(records)


def _selector_config() -> CandidatePoolSequenceSelectorConfig:
    return CandidatePoolSequenceSelectorConfig(
        group_column="candidate_branch",
        include_full_pool=False,
        include_leave_one_out=True,
        max_leave_one_out=8,
        min_group_frame_fraction=0.0,
        restore_missing_frames=False,
    )


def test_sequence_pool_selector_rejects_colliding_serialized_pool_labels() -> None:
    candidates = _candidate_rows(("raw/a", "raw a"))

    with pytest.raises(ValueError, match="candidate-pool labels collide"):
        build_sequence_candidate_pool_variants(candidates, config=_selector_config())


def test_sequence_pool_selector_keeps_distinct_serialized_pool_labels() -> None:
    candidates = _candidate_rows(("raw/a", "translated"))

    pools = build_sequence_candidate_pool_variants(candidates, config=_selector_config())

    assert set(pools) == {
        "without_candidate_branch_raw_a",
        "without_candidate_branch_translated",
    }
