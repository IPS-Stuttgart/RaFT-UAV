from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_reservoir import build_candidate_reservoir
from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["finite-best", "corrupt-inf"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 20.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "ranker_score": [0.9, np.inf],
            "confidence": [0.9, 0.1],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


def test_candidate_reservoir_falls_back_from_nonfinite_primary_score() -> None:
    reservoir = build_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=0,
            max_candidates_per_frame=1,
            score_column="ranker_score",
            fallback_score_column="confidence",
        ),
    )

    assert reservoir["track_id"].tolist() == ["finite-best"]
    assert reservoir.loc[0, "candidate_reservoir_score"] == 0.9


def test_oracle_recall_demotes_nonfinite_precomputed_scores() -> None:
    reservoir = _candidate_rows().rename(
        columns={"ranker_score": "candidate_reservoir_score"}
    )

    frame_rows, pooled, _by_sequence = build_oracle_recall_tables(
        reservoir,
        _truth_rows(),
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert frame_rows.loc[0, "oracle_top1_3d_m"] == 0.0
    assert pooled.loc[0, "oracle_top1_3d_m_mse"] == 0.0
