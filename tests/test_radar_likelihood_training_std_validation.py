from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import radar_likelihood_training as training


@pytest.mark.parametrize("field", ["radar_xy_std_m", "radar_z_std_m"])
@pytest.mark.parametrize(
    "std",
    [
        pytest.param(True, id="python-bool"),
        pytest.param(np.bool_(False), id="numpy-bool"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(np.array([1.0]), id="non-scalar-array"),
        pytest.param(np.ma.masked, id="masked"),
    ],
)
def test_training_collector_rejects_invalid_radar_standard_deviations(
    field: str,
    std: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        training.collect_radar_association_training_frame(
            rf_measurements=[],
            radar=pd.DataFrame(),
            truth=pd.DataFrame(),
            **{field: std},
        )


def test_training_collector_normalizes_radar_standard_deviations(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_collect(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(
        training,
        "_ORIGINAL_COLLECT_RADAR_ASSOCIATION_TRAINING_FRAME",
        fake_collect,
    )

    result = training.collect_radar_association_training_frame(
        rf_measurements=[],
        radar=pd.DataFrame(),
        truth=pd.DataFrame(),
        radar_xy_std_m=np.array(12.5),
        radar_z_std_m="35.5",
    )

    assert result.empty
    assert captured["radar_xy_std_m"] == 12.5
    assert captured["radar_z_std_m"] == 35.5
    assert isinstance(captured["radar_xy_std_m"], float)
    assert isinstance(captured["radar_z_std_m"], float)
