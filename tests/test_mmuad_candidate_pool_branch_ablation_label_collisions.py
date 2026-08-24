from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pool_branch_ablation import (
    build_candidate_pool_branch_ablation_pools,
)


def test_branch_ablation_rejects_colliding_normalized_pool_labels() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_branch": ["raw/a", "raw a"],
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="candidate_branch values collide after pool-label normalization",
    ):
        build_candidate_pool_branch_ablation_pools(candidates)


def test_branch_ablation_allows_collisions_when_group_pools_are_disabled() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_branch": ["raw/a", "raw a"],
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
        }
    )

    pools, manifest = build_candidate_pool_branch_ablation_pools(
        candidates,
        include_leave_one_out=False,
        include_only_one=False,
    )

    assert set(pools) == {"full_pool"}
    assert manifest["pool_label"].tolist() == ["full_pool"]
