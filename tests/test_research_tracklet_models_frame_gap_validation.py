from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.research.tracklet_models as tracklet_models


@pytest.mark.parametrize(
    "max_frame_gap",
    [
        -0.01,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        -np.inf,
        1.0 + 0.0j,
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_tracklet_feature_frame_rejects_invalid_frame_gaps(
    max_frame_gap: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_frame_gap must be a finite non-negative real scalar",
    ):
        tracklet_models.tracklet_feature_frame(
            pd.DataFrame(),
            max_frame_gap=max_frame_gap,
        )


def test_tracklet_feature_frame_rejects_boxed_boolean_gap() -> None:
    boxed = np.empty((), dtype=object)
    boxed[()] = np.bool_(True)

    with pytest.raises(
        ValueError,
        match="max_frame_gap must be a finite non-negative real scalar",
    ):
        tracklet_models.tracklet_feature_frame(pd.DataFrame(), max_frame_gap=boxed)


def test_tracklet_feature_frame_rejects_cyclic_boxed_gap() -> None:
    cyclic = np.empty((), dtype=object)
    cyclic[()] = cyclic

    with pytest.raises(
        ValueError,
        match="max_frame_gap must be a finite non-negative real scalar",
    ):
        tracklet_models.tracklet_feature_frame(pd.DataFrame(), max_frame_gap=cyclic)


def test_tracklet_feature_frame_normalizes_boxed_real_gap(monkeypatch) -> None:
    observed: dict[str, float] = {}

    def fake_tracklet_feature_frame(
        radar: pd.DataFrame,
        *,
        max_frame_gap: float,
    ) -> pd.DataFrame:
        observed["max_frame_gap"] = max_frame_gap
        return radar.copy()

    monkeypatch.setattr(
        tracklet_models,
        "_ORIGINAL_TRACKLET_FEATURE_FRAME",
        fake_tracklet_feature_frame,
    )
    boxed = np.empty((), dtype=object)
    boxed[()] = np.array(1.25)

    result = tracklet_models.tracklet_feature_frame(
        pd.DataFrame(),
        max_frame_gap=boxed,
    )

    assert result.empty
    assert observed["max_frame_gap"] == pytest.approx(1.25)
