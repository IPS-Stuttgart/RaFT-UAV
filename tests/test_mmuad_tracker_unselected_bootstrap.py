from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker


@pytest.mark.parametrize("bootstrap_flag", [False, np.bool_(False)])
def test_disabling_selected_bootstrap_starts_from_earliest_event(
    bootstrap_flag: object,
) -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq", "seq", "seq"],
                "time_s": [0.0, 10.0, 20.0],
                "source": ["candidate", "radar", "radar"],
                "track_id": ["singleton", "target", "target"],
                "x_m": [0.0, 100.0, 110.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [0.0, 0.0, 0.0],
            }
        )
    )
    config = TrackerConfig(
        first_selected_bootstrap=bootstrap_flag,
        selection_mobility_radius_m=0.0,
    )

    output = run_mmuad_tracker(candidates, config=config)

    assert output.selected_tracklets["time_s"].tolist() == [10.0, 20.0]
    first_estimate = output.estimates.iloc[0]
    assert first_estimate["time_s"] == 0.0
    assert not bool(first_estimate["selected_path_update"])
    assert first_estimate["update_action"] == "soft_anchor"
    assert first_estimate["state_x_m"] == 0.0


@pytest.mark.parametrize(
    "bootstrap_flag",
    [
        "False",
        "True",
        0,
        1,
        None,
        np.array(False),
        np.array([False]),
        np.ma.masked,
    ],
)
def test_tracker_rejects_ambiguous_selected_bootstrap_flag(
    bootstrap_flag: object,
) -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq"],
                "time_s": [0.0],
                "source": ["radar"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [0.0],
            }
        )
    )
    config = TrackerConfig(first_selected_bootstrap=bootstrap_flag)

    with pytest.raises(
        ValueError,
        match="first_selected_bootstrap must be a Boolean scalar",
    ):
        run_mmuad_tracker(candidates, config=config)
