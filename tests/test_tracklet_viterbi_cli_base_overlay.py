from __future__ import annotations

import pandas as pd
import pytest

from raft_uav import tracklet_viterbi_cli
from raft_uav.baselines.tracklet_viterbi import (
    run_async_cv_baseline_with_tracklet_viterbi_association,
)


def test_base_variant_accepts_canonical_cli_config_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(tracklet_viterbi_cli._TRACKLET_VARIANT_ENV, "base")
    monkeypatch.setenv(tracklet_viterbi_cli._TRACKLET_REPLAY_TRACKER_ENV, "cv")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()
    assert isinstance(config, tracklet_viterbi_cli._TrackletConfigOverlay)

    runner = tracklet_viterbi_cli._tracklet_runner_from_environment()
    records, selected = runner(
        rf_measurements=[],
        radar=pd.DataFrame(),
        config=config,
    )

    assert records == []
    assert selected.empty


def test_direct_base_runner_still_rejects_malformed_explicit_config() -> None:
    with pytest.raises(ValueError, match="TrackletViterbiAssociationConfig"):
        run_async_cv_baseline_with_tracklet_viterbi_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            config=False,
        )
