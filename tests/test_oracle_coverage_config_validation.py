from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_candidate_coverage import (
    build_oracle_candidate_coverage_diagnostics,
)
from raft_uav.evaluation.oracle_coverage import build_oracle_candidate_coverage


def _empty_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["time_s", "east_m", "north_m", "up_m"]
    return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)


_BUILDERS: tuple[Callable[..., Any], ...] = (
    build_oracle_candidate_coverage,
    build_oracle_candidate_coverage_diagnostics,
)


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(build_oracle_candidate_coverage, id="coverage"),
        pytest.param(
            build_oracle_candidate_coverage_diagnostics,
            id="diagnostics",
        ),
    ],
)
@pytest.mark.parametrize(
    "invalid_config",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param({}, id="empty-mapping"),
        pytest.param([], id="empty-sequence"),
    ],
)
def test_oracle_coverage_rejects_invalid_explicit_tracklet_configs(
    builder: Callable[..., Any],
    invalid_config: object,
) -> None:
    radar, truth = _empty_inputs()

    with pytest.raises(
        ValueError,
        match=(
            "^config must be a TrackletViterbiAssociationConfig instance or None$"
        ),
    ):
        builder(radar=radar, truth=truth, config=invalid_config)


@pytest.mark.parametrize("builder", _BUILDERS)
def test_oracle_coverage_accepts_explicit_tracklet_config(
    builder: Callable[..., Any],
) -> None:
    radar, truth = _empty_inputs()

    result = builder(
        radar=radar,
        truth=truth,
        config=TrackletViterbiAssociationConfig(max_candidates_per_frame=3),
    )

    if isinstance(result, tuple):
        report, summary = result
        assert report.empty
        assert summary["radar_frame_count"] == 0
    else:
        assert result.frame_coverage.empty
        assert result.summary["radar_frames"] == 0
