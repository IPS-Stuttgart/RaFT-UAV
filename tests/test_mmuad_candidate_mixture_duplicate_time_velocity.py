from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
)


def _linear_candidates_and_duplicate_template() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["lidar_360", "lidar_360", "lidar_360"],
            "track_id": ["t0", "t1", "t2"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [1.0, 1.0, 1.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
        }
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 1.0, 1.0, 2.0],
        }
    )
    return candidates, template


def test_candidate_mixture_duplicate_target_times_keep_velocities_finite() -> None:
    candidates, template = _linear_candidates_and_duplicate_template()

    result = run_candidate_mixture_map(
        candidates,
        target_template=template,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=0.0,
            target_time_tolerance_s=0.01,
            iterations=1,
        ),
    )

    assert result.estimates["time_s"].tolist() == [0.0, 1.0, 1.0, 2.0]
    velocity = result.estimates[["v_x_mps", "v_y_mps", "v_z_mps"]].to_numpy(float)
    assert np.isfinite(velocity).all()
    assert velocity[:, 0] == pytest.approx(np.ones(4))
    assert velocity[:, 1:] == pytest.approx(np.zeros((4, 2)))


def test_duplicate_target_times_do_not_create_artificial_acceleration() -> None:
    candidates, template = _linear_candidates_and_duplicate_template()

    result = run_candidate_mixture_map(
        candidates,
        target_template=template,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=7200.0,
            target_time_tolerance_s=0.01,
            iterations=1,
        ),
    )

    position = result.estimates[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    expected = np.column_stack(
        [
            np.array([0.0, 1.0, 1.0, 2.0]),
            np.zeros(4),
            np.zeros(4),
        ]
    )
    assert position == pytest.approx(expected, abs=2.0e-5)
    assert result.estimates["v_x_mps"].to_numpy(float) == pytest.approx(
        np.ones(4),
        abs=2.0e-5,
    )
