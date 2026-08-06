from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import (
    add_temporal_candidate_consensus,
)
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
)


_AUGMENTERS: tuple[Callable[..., object], ...] = (
    add_temporal_candidate_consensus,
    add_assignment_temporal_candidate_consensus,
)


def _boxed(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


_MALFORMED_SCORES = (
    pytest.param(True, id="python-bool"),
    pytest.param(np.bool_(False), id="numpy-bool"),
    pytest.param(1.0 + 2.0j, id="complex"),
    pytest.param(_boxed(True), id="boxed-bool"),
    pytest.param(_boxed(np.array([0.9])), id="boxed-vector"),
    pytest.param(_boxed(np.ma.array(0.9, mask=True)), id="boxed-masked"),
)


@pytest.mark.parametrize("augment", _AUGMENTERS)
def test_temporal_consensus_falls_back_from_nonfinite_primary_scores(
    augment: Callable[..., object],
) -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [0.0] * 3,
            "source": ["positive_inf", "finite", "negative_inf"],
            "track_id": ["positive_inf", "finite", "negative_inf"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0] * 3,
            "z_m": [0.0] * 3,
            "ranker_score": [np.inf, 0.5, -np.inf],
            "confidence": [0.2, 0.5, 0.8],
        }
    )

    augmented = augment(candidates)
    rows = augmented.rows.set_index("track_id")
    base = rows["candidate_reservoir_temporal_base_score"]
    consensus = rows["candidate_temporal_consensus_score"]

    assert np.isfinite(base.to_numpy(float)).all()
    assert np.isfinite(consensus.to_numpy(float)).all()
    assert base.loc["positive_inf"] == pytest.approx(0.0)
    assert base.loc["finite"] == pytest.approx(0.5)
    assert base.loc["negative_inf"] == pytest.approx(1.0)


@pytest.mark.parametrize("augment", _AUGMENTERS)
@pytest.mark.parametrize("malformed_score", _MALFORMED_SCORES)
def test_temporal_consensus_ignores_malformed_primary_scores(
    augment: Callable[..., object],
    malformed_score: object,
) -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [0.0] * 3,
            "source": ["malformed", "finite", "high"],
            "track_id": ["malformed", "finite", "high"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0] * 3,
            "z_m": [0.0] * 3,
            "ranker_score": pd.Series(
                [malformed_score, 0.5, 0.8],
                dtype=object,
            ),
            "confidence": [0.2, 0.5, 0.8],
        }
    )

    augmented = augment(candidates)
    base = augmented.rows.set_index("track_id")[
        "candidate_reservoir_temporal_base_score"
    ]

    assert np.isfinite(base.to_numpy(float)).all()
    assert base.loc["malformed"] == pytest.approx(0.0)
    assert base.loc["finite"] == pytest.approx(0.5)
    assert base.loc["high"] == pytest.approx(1.0)


@pytest.mark.parametrize("augment", _AUGMENTERS)
def test_temporal_consensus_preserves_recursively_boxed_real_scores(
    augment: Callable[..., object],
) -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [0.0] * 3,
            "source": ["boxed", "finite", "high"],
            "track_id": ["boxed", "finite", "high"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0] * 3,
            "z_m": [0.0] * 3,
            "ranker_score": pd.Series(
                [_boxed(np.array(0.2)), 0.5, 0.8],
                dtype=object,
            ),
            "confidence": [0.9, 0.5, 0.8],
        }
    )

    augmented = augment(candidates)
    base = augmented.rows.set_index("track_id")[
        "candidate_reservoir_temporal_base_score"
    ]

    assert base.loc["boxed"] == pytest.approx(0.0)
    assert base.loc["finite"] == pytest.approx(0.5)
    assert base.loc["high"] == pytest.approx(1.0)
