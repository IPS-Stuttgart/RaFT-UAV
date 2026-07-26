from __future__ import annotations

import pandas as pd

from raft_uav.evaluation.oracle_coverage import _rank_by_catprob


def _rank_for_oracle(rows: pd.DataFrame) -> float:
    oracle_position = int(rows.index[rows["track_id"].eq("oracle")][0])
    return _rank_by_catprob(rows, oracle_position)


def test_oracle_catprob_rank_is_invariant_to_equal_score_row_order() -> None:
    candidates = pd.DataFrame(
        {
            "track_id": ["higher", "oracle", "peer"],
            "cat_prob_uav": [0.9, 0.5, 0.5],
        }
    )
    reversed_candidates = candidates.iloc[::-1].reset_index(drop=True)

    assert _rank_for_oracle(candidates) == 2.5
    assert _rank_for_oracle(reversed_candidates) == 2.5
