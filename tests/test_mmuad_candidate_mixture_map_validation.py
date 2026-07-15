from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map import run_candidate_mixture_map


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_sigma_m", np.nan),
        ("default_sigma_m", np.inf),
        ("sigma_max_m", np.inf),
        ("score_weight", np.nan),
        ("score_weight", np.inf),
        ("temperature", np.nan),
        ("temperature", np.inf),
        ("sigma_log_weight", np.nan),
        ("sigma_log_weight", -np.inf),
        ("huber_delta", np.nan),
        ("huber_delta", np.inf),
        ("smoothness_weight", np.nan),
        ("smoothness_weight", np.inf),
        ("anchor_weight", np.nan),
        ("anchor_weight", np.inf),
        ("tolerance_m", np.nan),
        ("tolerance_m", np.inf),
        ("target_time_tolerance_s", np.nan),
        ("target_time_tolerance_s", np.inf),
        ("min_measurement_precision", np.nan),
        ("max_measurement_precision", np.inf),
    ],
)
def test_candidate_mixture_rejects_nonfinite_controls(
    field: str,
    value: object,
) -> None:
    config = replace(CandidateMixtureMapConfig(), **{field: value})

    with pytest.raises(ValueError, match=field):
        run_candidate_mixture_map(pd.DataFrame(), config=config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 1.5),
        ("top_k", True),
        ("top_k", np.array([1])),
        ("iterations", 1.5),
        ("iterations", True),
        ("iterations", np.array([1])),
    ],
)
def test_candidate_mixture_rejects_noninteger_controls(
    field: str,
    value: object,
) -> None:
    config = replace(CandidateMixtureMapConfig(), **{field: value})

    with pytest.raises(ValueError, match=field):
        run_candidate_mixture_map(pd.DataFrame(), config=config)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_candidate_mixture_rejects_invalid_precision_bounds(
    minimum: float,
    maximum: float,
) -> None:
    config = replace(
        CandidateMixtureMapConfig(),
        min_measurement_precision=minimum,
        max_measurement_precision=maximum,
    )

    with pytest.raises(ValueError, match="measurement precision bounds"):
        run_candidate_mixture_map(pd.DataFrame(), config=config)


def test_candidate_mixture_accepts_finite_numpy_scalars() -> None:
    config = replace(
        CandidateMixtureMapConfig(),
        top_k=np.int64(0),
        iterations=np.float64(1.0),
        score_weight=np.float32(1.0),
        min_measurement_precision=np.float64(1.0e-6),
        max_measurement_precision=np.array(1.0e6),
    )

    result = run_candidate_mixture_map(pd.DataFrame(), config=config)

    assert result.estimates.empty
    assert result.assignments.empty
