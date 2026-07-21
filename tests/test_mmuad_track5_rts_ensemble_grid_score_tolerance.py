from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.track5_rts_ensemble_grid as rts_grid


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        True,
        1.0 + 0.0j,
        np.asarray([0.1]),
    ],
)
def test_rts_grid_rejects_invalid_score_time_tolerance(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="score_time_tolerance_s must be a non-negative finite scalar",
    ):
        rts_grid.run_track5_rts_ensemble_grid_search(
            [],
            template=pd.DataFrame(),
            truth=pd.DataFrame(),
            score_time_tolerance_s=value,
        )


def test_rts_grid_normalizes_zero_dimensional_score_tolerance(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object):
        captured.update(kwargs)
        return "grid", {"best": {}}

    monkeypatch.setattr(rts_grid, "_ORIGINAL_RUN_GRID_SEARCH", fake_run)

    result = rts_grid.run_track5_rts_ensemble_grid_search(
        [],
        template=pd.DataFrame(),
        truth=pd.DataFrame(),
        score_time_tolerance_s=np.asarray(0.0),
    )

    assert result == ("grid", {"best": {}})
    assert captured["score_time_tolerance_s"] == 0.0
    assert isinstance(captured["score_time_tolerance_s"], float)
