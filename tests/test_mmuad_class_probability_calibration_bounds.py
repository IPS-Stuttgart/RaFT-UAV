from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.class_probability_calibration as calibration


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["0000", "0001", "0002", "0003"],
            "class_prob_0": [0.9, 0.8, 0.2, 0.1],
            "class_prob_1": [0.1, 0.2, 0.8, 0.9],
        }
    )


def _labels() -> dict[str, str]:
    return {"0000": "0", "0001": "0", "0002": "1", "0003": "1"}


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (0.0, 1.0, "min_temperature"),
        (-1.0, 1.0, "min_temperature"),
        (float("nan"), 1.0, "min_temperature"),
        (0.1, 0.0, "max_temperature"),
        (0.1, float("inf"), "max_temperature"),
        (2.0, 1.0, "less than"),
        (1.0, 1.0, "less than"),
    ],
)
def test_fit_rejects_invalid_temperature_bounds(
    minimum: float,
    maximum: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calibration.fit_temperature_calibrator(
            _predictions(),
            _labels(),
            min_temperature=minimum,
            max_temperature=maximum,
        )


def test_fit_rejects_interval_below_numerical_temperature_floor() -> None:
    with pytest.raises(ValueError, match="effective numerical lower bound"):
        calibration.fit_temperature_calibrator(
            _predictions(),
            _labels(),
            min_temperature=1.0e-6,
            max_temperature=5.0e-5,
        )


def test_optimizer_failure_fallback_stays_within_requested_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "minimize_scalar",
        lambda *args, **kwargs: SimpleNamespace(
            success=False,
            x=np.nan,
            message="forced optimizer failure",
        ),
    )

    model, summary = calibration.fit_temperature_calibrator(
        _predictions(),
        _labels(),
        min_temperature=2.0,
        max_temperature=3.0,
    )

    assert model.temperature == 2.0
    assert summary["optimizer_success"] is False
