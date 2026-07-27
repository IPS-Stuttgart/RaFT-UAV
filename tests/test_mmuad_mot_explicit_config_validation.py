from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from raft_uav.mmuad import run_mmuad_multi_object_tracker as package_runner
from raft_uav.mmuad.mot import run_mmuad_multi_object_tracker as module_runner
from raft_uav.mmuad.schema import CandidateFrame


def _empty_candidates() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            columns=["sequence_id", "time_s", "source", "x_m", "y_m", "z_m"]
        )
    )


@pytest.mark.parametrize(
    "runner",
    [
        pytest.param(module_runner, id="module"),
        pytest.param(package_runner, id="package"),
    ],
)
@pytest.mark.parametrize("invalid_config", [False, 0, "", {}, []])
def test_mot_rejects_invalid_explicit_config_before_empty_return(
    runner: Callable[..., Any],
    invalid_config: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="^config must be a MultiObjectTrackerConfig or None$",
    ):
        runner(_empty_candidates(), config=invalid_config)


def test_mot_preserves_none_default_for_empty_inputs() -> None:
    output = module_runner(_empty_candidates(), config=None)

    assert output.estimates.empty
    assert output.selected_tracklets.empty
    assert output.metrics == {
        "sequences": {},
        "pooled": {"count": 0, "track_count": 0},
    }
