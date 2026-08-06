from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


@dataclass(frozen=True)
class RFMeasurement:
    time_s: float
    vector: np.ndarray
    sequence_id: object = None


def _rf(sequence_id: object) -> RFMeasurement:
    return RFMeasurement(
        time_s=0.0,
        vector=np.array([1.0, 2.0, 3.0]),
        sequence_id=sequence_id,
    )


def test_delayed_initialization_rejects_multiple_radar_sequences() -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a", "flight-b"]})

    with pytest.raises(ValueError, match="inputs from one sequence"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-a")],
            radar=radar,
        )


def test_delayed_initialization_rejects_multiple_rf_sequences() -> None:
    with pytest.raises(ValueError, match="inputs from one sequence"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-a"), _rf("flight-b")],
            radar=pd.DataFrame(),
        )


def test_delayed_initialization_rejects_cross_modality_sequence_mismatch() -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a"]})

    with pytest.raises(ValueError, match="inputs from one sequence"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-b")],
            radar=radar,
        )


def test_delayed_initialization_allows_missing_sequence_labels() -> None:
    radar = pd.DataFrame(
        {"sequence_id": ["flight-a", None, pd.NA, np.nan, ""]}
    )
    rf_measurements = (
        measurement
        for measurement in [_rf("flight-a"), _rf(None), _rf("")]
    )

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=rf_measurements,
        radar=radar,
    )

    assert len(hypotheses) == 3
