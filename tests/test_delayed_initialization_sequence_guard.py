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
    flight_id: object = None


def _rf(
    sequence_id: object = None,
    *,
    flight_id: object = None,
) -> RFMeasurement:
    return RFMeasurement(
        time_s=0.0,
        vector=np.array([1.0, 2.0, 3.0]),
        sequence_id=sequence_id,
        flight_id=flight_id,
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


def test_delayed_initialization_rejects_partial_radar_sequence_labels() -> None:
    radar = pd.DataFrame({"sequence_id": ["flight-a", None]})

    with pytest.raises(ValueError, match="complete sequence metadata"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-a")],
            radar=radar,
        )


def test_delayed_initialization_rejects_partial_rf_sequence_labels() -> None:
    with pytest.raises(ValueError, match="complete sequence metadata"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-a"), _rf(None)],
            radar=pd.DataFrame(),
        )


def test_delayed_initialization_rejects_flight_id_mismatch() -> None:
    radar = pd.DataFrame({"flight_id": ["flight-a"]})

    with pytest.raises(ValueError, match="inputs from one sequence"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf(flight_id="flight-b")],
            radar=radar,
        )


def test_delayed_initialization_rejects_conflicting_sequence_aliases() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": ["flight-a"],
            "flight_id": ["flight-b"],
        }
    )

    with pytest.raises(ValueError, match="consistent sequence aliases"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf("flight-a")],
            radar=radar,
        )


def test_delayed_initialization_rejects_non_scalar_sequence_ids() -> None:
    radar = pd.DataFrame({"sequence_id": [["flight-a"]]})

    with pytest.raises(ValueError, match="sequence identifiers must be scalar"):
        build_delayed_initial_hypotheses(
            rf_measurements=[_rf()],
            radar=radar,
        )


def test_delayed_initialization_allows_fully_unlabeled_legacy_inputs() -> None:
    radar = pd.DataFrame(
        {
            "sequence_id": [
                None,
                pd.NA,
                np.nan,
                "",
                "nan",
                "None",
                "<NA>",
                "NaT",
            ]
        }
    )
    rf_measurements = [_rf(None), _rf("nan"), _rf("<NA>")]

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=rf_measurements,
        radar=radar,
    )

    assert len(hypotheses) == 3


def test_delayed_initialization_allows_one_fully_unlabeled_modality() -> None:
    radar = pd.DataFrame({"sequence_id": [" flight-a "]})

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[_rf()],
        radar=radar,
    )

    assert len(hypotheses) == 1


def test_delayed_initialization_accepts_matching_sequence_aliases() -> None:
    radar = pd.DataFrame({"flight_id": [" flight-a "]})

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[_rf("flight-a")],
        radar=radar,
    )

    assert len(hypotheses) == 1
