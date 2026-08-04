from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


_INVALID_SEQUENCE_IDS = [None, np.nan, pd.NA, "", "   ", "NaN", "None", "<NA>"]
_INVALID_SEQUENCE_ID_NAMES = [
    "none",
    "nan",
    "pandas-na",
    "empty",
    "whitespace",
    "nan-string",
    "none-string",
    "pandas-na-string",
]


def _estimates(sequence_id: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "time_s": [0.0],
            "output_track_id": ["prediction"],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )


def _truth(sequence_id: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id],
            "time_s": [0.0],
            "track_id": ["object"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


@pytest.mark.parametrize(
    "sequence_id",
    _INVALID_SEQUENCE_IDS,
    ids=_INVALID_SEQUENCE_ID_NAMES,
)
def test_mot_metrics_reject_missing_estimate_sequence_ids(sequence_id: object) -> None:
    with pytest.raises(
        ValueError,
        match="estimates sequence_id values must be non-missing",
    ):
        compute_multi_object_metrics(
            _estimates(sequence_id),
            _truth("seqA"),
            match_distance_m=1.0,
        )


@pytest.mark.parametrize(
    "sequence_id",
    _INVALID_SEQUENCE_IDS,
    ids=_INVALID_SEQUENCE_ID_NAMES,
)
def test_mot_metrics_reject_missing_truth_sequence_ids(sequence_id: object) -> None:
    with pytest.raises(
        ValueError,
        match="truth sequence_id values must be non-missing",
    ):
        compute_multi_object_metrics(
            _estimates("seqA"),
            _truth(sequence_id),
            match_distance_m=1.0,
        )


def test_mot_metrics_reject_missing_sequence_ids_without_truth() -> None:
    with pytest.raises(
        ValueError,
        match="estimates sequence_id values must be non-missing",
    ):
        compute_multi_object_metrics(_estimates(None), None)


def test_mot_metrics_keep_numeric_zero_sequence_id() -> None:
    metrics = compute_multi_object_metrics(
        _estimates(0),
        _truth(0),
        match_distance_m=1.0,
    )

    assert metrics["matches"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
