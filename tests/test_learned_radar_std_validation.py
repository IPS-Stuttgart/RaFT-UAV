import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import learned_radar_association as learned_assoc
from raft_uav.baselines import radar_association
from raft_uav.baselines import stateful_learned_radar_association as stateful_assoc
from raft_uav.baselines.learned_radar_likelihood import LearnedRadarAssociationModel


def _model() -> LearnedRadarAssociationModel:
    return LearnedRadarAssociationModel(
        feature_names=("cat_prob_uav",),
        mean=np.array([0.0]),
        scale=np.array([1.0]),
        weights=np.array([1.0]),
        intercept=0.0,
    )


_RUNNERS = (
    learned_assoc.run_async_cv_baseline_with_learned_radar_association,
    stateful_assoc.run_async_cv_baseline_with_stateful_learned_radar_association,
)


@pytest.mark.parametrize("runner", _RUNNERS, ids=("per-frame", "stateful"))
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
def test_learned_radar_runners_reject_invalid_standard_deviations(
    runner,
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_events(*args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        pytest.fail("standard-deviation validation must run before event construction")

    monkeypatch.setattr(learned_assoc, "_events", unexpected_events)
    monkeypatch.setattr(radar_association, "_events", unexpected_events)
    kwargs = {
        "rf_measurements": [],
        "radar": pd.DataFrame(),
        "model": _model(),
        field: value,
    }

    with pytest.raises(
        ValueError,
        match=rf"^{field} must be a finite positive real scalar$",
    ):
        runner(**kwargs)


@pytest.mark.parametrize("runner", _RUNNERS, ids=("per-frame", "stateful"))
def test_learned_radar_runners_accept_zero_dimensional_real_standard_deviations(
    runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(learned_assoc, "_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(radar_association, "_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(learned_assoc, "_empty_selected_radar", lambda frame: frame.copy())
    monkeypatch.setattr(radar_association, "_empty_selected_radar", lambda frame: frame.copy())

    records, selected = runner(
        rf_measurements=[],
        radar=pd.DataFrame(),
        model=_model(),
        radar_xy_std_m=np.array(25.0),
        radar_z_std_m=np.ma.array(35.0, mask=False),
    )

    assert records == []
    assert selected.empty
