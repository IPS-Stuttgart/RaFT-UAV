from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)


def _equal_score_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["a-near", "b-far"],
            "candidate_branch": ["near", "far"],
            "x_m": [0.0, 10.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [0.5, 0.5],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


def test_equal_score_oracle_attribution_is_independent_of_row_order() -> None:
    candidates = _equal_score_candidates()
    forward, _, _, _ = build_candidate_oracle_attribution_tables(
        candidates,
        _truth(),
        top_k_values=(1,),
    )
    reversed_rows, _, _, _ = build_candidate_oracle_attribution_tables(
        candidates.iloc[::-1].reset_index(drop=True),
        _truth(),
        top_k_values=(1,),
    )

    columns = [
        "oracle_all_candidate_track_id",
        "oracle_all_candidate_branch",
        "oracle_all_rank",
        "oracle_in_top1",
        "oracle_top1_3d_m",
    ]
    assert forward[columns].to_dict(orient="records") == reversed_rows[
        columns
    ].to_dict(orient="records")
    assert forward.loc[0, "oracle_all_candidate_track_id"] == "a-near"
    assert forward.loc[0, "oracle_all_rank"] == 1
    assert bool(forward.loc[0, "oracle_in_top1"])
    assert forward.loc[0, "oracle_top1_3d_m"] == 0.0
