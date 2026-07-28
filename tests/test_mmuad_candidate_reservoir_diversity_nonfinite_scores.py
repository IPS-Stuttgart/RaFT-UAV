from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir_diversity import diversity_cap_reservoir


def _candidate_rows(
    *,
    invalid_primary: float,
    invalid_fallback: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["invalid", "finite"],
            "candidate_branch": ["raw", "raw"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [invalid_primary, 0.8],
            "confidence": [invalid_fallback, 0.8],
        }
    )


def _selected_track(rows: pd.DataFrame) -> str:
    capped = diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=1,
        min_per_source=0,
        min_per_branch=0,
    )
    assert len(capped) == 1
    return str(capped.iloc[0]["track_id"])


def test_diversity_cap_uses_fallback_for_nonfinite_primary_score() -> None:
    rows = _candidate_rows(invalid_primary=np.inf, invalid_fallback=0.1)

    assert _selected_track(rows) == "finite"


def test_diversity_cap_neutralizes_nonfinite_fallback_score() -> None:
    rows = _candidate_rows(invalid_primary=np.nan, invalid_fallback=np.inf)

    assert _selected_track(rows) == "finite"
