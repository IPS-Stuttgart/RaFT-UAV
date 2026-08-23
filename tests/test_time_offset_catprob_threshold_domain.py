from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.time_offset import (
    catprob_candidate_pool,
    run_time_offset_diagnostic,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat_prob_uav": [0.9, 0.4],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_catprob_candidate_pool_rejects_out_of_range_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        catprob_candidate_pool(_candidates(), threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_catprob_candidate_pool_accepts_probability_boundaries(threshold: float) -> None:
    selected = catprob_candidate_pool(_candidates(), threshold)

    assert not selected.empty


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_time_offset_runner_rejects_invalid_catprob_before_data_access(
    tmp_path,
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="radar_catprob_threshold must be between 0 and 1",
    ):
        run_time_offset_diagnostic(
            dataset_root=tmp_path / "missing-dataset",
            flight_name="missing-flight",
            source="radar",
            tau_min_s=-1.0,
            tau_max_s=1.0,
            tau_step_s=0.1,
            radar_catprob_threshold=threshold,
        )
