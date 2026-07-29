from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_spatial import (
    spatial_diversity_cap_reservoir,
)


def _candidate_rows(
    *,
    invalid_primary: object = np.inf,
    invalid_fallback: object = 0.1,
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
    capped = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=1,
        min_per_source=0,
        min_per_branch=0,
        spatial_diversity_weight=0.0,
    )
    assert len(capped) == 1
    return str(capped.iloc[0]["track_id"])


def test_spatial_cap_uses_fallback_for_nonfinite_primary_score() -> None:
    assert _selected_track(_candidate_rows()) == "finite"


def test_spatial_cap_neutralizes_nonfinite_fallback_score() -> None:
    rows = _candidate_rows(invalid_primary=np.nan, invalid_fallback=np.inf)

    assert _selected_track(rows) == "finite"


def test_spatial_cap_rejects_complex_score_without_losing_real_rows() -> None:
    rows = _candidate_rows(invalid_primary=1.0 + 2.0j, invalid_fallback=0.1)
    rows["candidate_reservoir_score"] = pd.Series(
        [1.0 + 2.0j, 0.8 + 0.0j],
        dtype=complex,
    )

    assert _selected_track(rows) == "finite"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_candidates_per_frame", True),
        ("max_candidates_per_frame", 1.5),
        ("max_candidates_per_frame", -1),
        ("max_candidates_per_frame", np.nan),
        ("min_per_source", np.bool_(False)),
        ("min_per_source", np.ma.masked),
        ("min_per_branch", np.array([1])),
        ("min_per_branch", np.inf),
        ("min_per_branch", "not-an-integer"),
    ],
)
def test_spatial_cap_rejects_invalid_integer_controls(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be a non-negative integer"):
        spatial_diversity_cap_reservoir(_candidate_rows(), **{name: value})


def test_spatial_cap_validates_controls_before_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="max_candidates_per_frame must be a non-negative integer",
    ):
        spatial_diversity_cap_reservoir(
            pd.DataFrame(),
            max_candidates_per_frame=2.5,
        )


def test_spatial_cap_accepts_exact_integer_equivalent_controls() -> None:
    capped = spatial_diversity_cap_reservoir(
        _candidate_rows(),
        max_candidates_per_frame=np.array(0),
        min_per_source="0",
        min_per_branch=np.float64(0.0),
    )

    assert len(capped) == 2
