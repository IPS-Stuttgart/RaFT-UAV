from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.calibration.bias import make_bias_training_examples


def _bias_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    return make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.0,
    )


def test_bias_examples_scope_shared_sequence_by_physical_flight() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    rows = _bias_examples(measurements, truth)

    assert rows["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert rows["bias_east_m"].tolist() == [1.0, 1.0]
    assert rows["bias_north_m"].tolist() == [0.0, 0.0]


def test_bias_examples_scope_flight_id_only_inputs() -> None:
    measurements = pd.DataFrame(
        {
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    rows = _bias_examples(measurements, truth)

    assert rows["flight_id"].tolist() == ["flight-a", "flight-b"]
    assert rows["bias_east_m"].tolist() == [1.0, 1.0]


def test_bias_examples_reject_one_sided_flight_subdivision() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="all disambiguating"):
        _bias_examples(measurements, truth)


def test_bias_examples_allow_unambiguous_one_sided_flight_metadata() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    rows = _bias_examples(measurements, truth)

    assert rows["sequence_id"].tolist() == ["seq-a", "seq-b"]
    assert rows["bias_east_m"].tolist() == [1.0, 1.0]


def test_bias_examples_reject_missing_scope_inside_pooled_inputs() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", None],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["shared", "shared"],
            "flight_id": ["flight-a", "flight-b"],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="every measurement row"):
        _bias_examples(measurements, truth)


def test_bias_examples_ignore_shared_scope_alias_that_is_entirely_missing() -> None:
    measurements = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "flight_id": [None, None],
            "time_s": [0.0, 0.0],
            "east_m": [11.0, 101.0],
            "north_m": [1.0, 10.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq-a", "seq-b"],
            "flight_id": [None, None],
            "time_s": [0.0, 0.0],
            "east_m": [10.0, 100.0],
            "north_m": [1.0, 10.0],
        }
    )

    rows = _bias_examples(measurements, truth)

    assert rows["sequence_id"].tolist() == ["seq-a", "seq-b"]
    assert rows["bias_east_m"].tolist() == [1.0, 1.0]
