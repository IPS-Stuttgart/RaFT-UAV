from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import radar_association
from raft_uav.baselines import run_async_cv_baseline_with_radar_association


@pytest.mark.parametrize("field", ("radar_xy_std_m", "radar_z_std_m"))
@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        1.0 + 1.0j,
        np.array([1.0]),
        np.array(1.0 + 1.0j, dtype=object),
        np.ma.masked,
    ),
)
def test_core_radar_runner_rejects_invalid_standard_deviations(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_events(*args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        pytest.fail("standard-deviation validation must run before event construction")

    monkeypatch.setattr(radar_association, "_events", unexpected_events)
    kwargs = {
        "rf_measurements": [],
        "radar": pd.DataFrame(),
        "association": "prediction-nis",
        field: value,
    }

    with pytest.raises(
        ValueError,
        match=rf"^{field} must be a finite positive real scalar$",
    ):
        run_async_cv_baseline_with_radar_association(**kwargs)


def test_core_radar_runner_accepts_zero_dimensional_real_standard_deviations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(radar_association, "_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        radar_association,
        "_empty_selected_radar",
        lambda frame: frame.copy(),
    )

    records, selected = run_async_cv_baseline_with_radar_association(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="prediction-nis",
        radar_xy_std_m=np.array(25.0),
        radar_z_std_m=np.ma.array(35.0, mask=False),
    )

    assert records == []
    assert selected.empty
