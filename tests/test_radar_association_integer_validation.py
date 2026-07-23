from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.baselines.radar_association as radar_association
from raft_uav.baselines.radar_association import (
    run_async_cv_baseline_with_radar_association,
)

_INTEGER_PARAMETERS = (
    "track_bank_max_hypotheses",
    "track_bank_max_assignments",
    "track_bank_max_candidates",
    "stable_segment_min_frames",
)


@pytest.mark.parametrize("parameter", _INTEGER_PARAMETERS)
@pytest.mark.parametrize(
    "value",
    [
        0,
        1.5,
        True,
        np.bool_(False),
        np.nan,
        np.array([2]),
        np.ma.masked,
    ],
)
def test_radar_association_rejects_malformed_positive_integer_controls(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        run_async_cv_baseline_with_radar_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            association="prediction-nis",
            **{parameter: value},
        )


@pytest.mark.parametrize("parameter", _INTEGER_PARAMETERS)
def test_radar_association_normalizes_scalar_like_integer_controls(
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> tuple[list[object], pd.DataFrame]:
        captured.update(kwargs)
        return [], pd.DataFrame()

    monkeypatch.setattr(
        radar_association,
        "_ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_RADAR_ASSOCIATION",
        fake_run,
    )

    run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="prediction-nis",
        **{parameter: "2"},
    )

    assert captured[parameter] == 2
    assert isinstance(captured[parameter], int)
