from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad import tracker


_ROW_POSITION = "__raft_uav_truth_error_row_position"


def _scoped_truth(
    *,
    sequence_id: str,
    flight_id: str,
    x_offset: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": [sequence_id, sequence_id],
            "flight_id": [flight_id, flight_id],
            "time_s": [0.0, 1.0],
            "x_m": [x_offset, x_offset + 10.0],
            "y_m": [x_offset + 1.0, x_offset + 11.0],
            "z_m": [x_offset + 2.0, x_offset + 12.0],
        }
    )


def _estimate_row(
    *,
    sequence_id: str,
    flight_id: str,
    time_s: float,
    x_m: float,
    y_m: float,
    z_m: float,
) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "flight_id": flight_id,
        "time_s": time_s,
        "state_x_m": x_m,
        "state_y_m": y_m,
        "state_z_m": z_m,
    }


def _pooled_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.concat(
        [
            _scoped_truth(
                sequence_id="seq-a",
                flight_id="shared-flight",
                x_offset=0.0,
            ),
            _scoped_truth(
                sequence_id="seq-b",
                flight_id="shared-flight",
                x_offset=100.0,
            ),
            _scoped_truth(
                sequence_id="seq-a",
                flight_id="other-flight",
                x_offset=200.0,
            ),
        ],
        ignore_index=True,
    )
    estimates = pd.DataFrame(
        [
            _estimate_row(
                sequence_id="seq-b",
                flight_id="shared-flight",
                time_s=0.5,
                x_m=105.0,
                y_m=106.0,
                z_m=107.0,
            ),
            _estimate_row(
                sequence_id="seq-a",
                flight_id="other-flight",
                time_s=0.5,
                x_m=205.0,
                y_m=206.0,
                z_m=207.0,
            ),
            _estimate_row(
                sequence_id="seq-a",
                flight_id="shared-flight",
                time_s=0.5,
                x_m=5.0,
                y_m=6.0,
                z_m=7.0,
            ),
        ],
        index=pd.Index([17, 4, 99], name="row_id"),
    )
    return estimates, truth


def _unique_flight_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.concat(
        [
            _scoped_truth(
                sequence_id="seq-a",
                flight_id="flight-a",
                x_offset=0.0,
            ),
            _scoped_truth(
                sequence_id="seq-b",
                flight_id="flight-b",
                x_offset=100.0,
            ),
        ],
        ignore_index=True,
    )
    estimates = pd.DataFrame(
        [
            _estimate_row(
                sequence_id="seq-a",
                flight_id="flight-a",
                time_s=0.5,
                x_m=5.0,
                y_m=6.0,
                z_m=7.0,
            ),
            _estimate_row(
                sequence_id="seq-b",
                flight_id="flight-b",
                time_s=0.5,
                x_m=105.0,
                y_m=106.0,
                z_m=107.0,
            ),
        ]
    )
    return estimates, truth


@pytest.mark.parametrize(
    "score_truth_errors",
    [tracker.add_truth_errors, tracker._LEGACY.add_truth_errors],
    ids=["public", "legacy"],
)
def test_add_truth_errors_stays_inside_sequence_and_flight_scope(
    score_truth_errors,
) -> None:
    estimates, truth = _pooled_inputs()

    scored = score_truth_errors(estimates, truth)

    assert scored.index.equals(estimates.index)
    assert scored["sequence_id"].tolist() == estimates["sequence_id"].tolist()
    assert scored["flight_id"].tolist() == estimates["flight_id"].tolist()
    assert np.allclose(scored["truth_x_m"], [105.0, 205.0, 5.0])
    assert np.allclose(scored["error_2d_m"], 0.0)
    assert np.allclose(scored["error_3d_m"], 0.0)


def test_add_truth_errors_preserves_existing_helper_named_column() -> None:
    estimates, truth = _pooled_inputs()
    expected = ["keep-17", "keep-4", "keep-99"]
    estimates[_ROW_POSITION] = expected

    scored = tracker.add_truth_errors(estimates, truth)

    assert scored[_ROW_POSITION].tolist() == expected


def test_add_truth_errors_rejects_ambiguous_one_sided_flight_scope() -> None:
    estimates = pd.DataFrame(
        [
            _estimate_row(
                sequence_id="seq",
                flight_id="flight-a",
                time_s=0.5,
                x_m=5.0,
                y_m=6.0,
                z_m=7.0,
            ),
            _estimate_row(
                sequence_id="seq",
                flight_id="flight-b",
                time_s=0.5,
                x_m=5.0,
                y_m=6.0,
                z_m=7.0,
            ),
        ]
    )
    truth = _scoped_truth(
        sequence_id="seq",
        flight_id="flight-a",
        x_offset=0.0,
    ).drop(columns="flight_id")

    with pytest.raises(ValueError, match="ambiguous flight_id metadata"):
        tracker.add_truth_errors(estimates, truth)


@pytest.mark.parametrize(
    "score_truth_errors",
    [tracker.add_truth_errors, tracker._LEGACY.add_truth_errors],
    ids=["public", "legacy"],
)
@pytest.mark.parametrize("missing_sequence_from", ["estimates", "truth"])
def test_add_truth_errors_rejects_ambiguous_one_sided_sequence_scope(
    score_truth_errors,
    missing_sequence_from: str,
) -> None:
    estimates, truth = _pooled_inputs()
    if missing_sequence_from == "estimates":
        estimates = estimates.drop(columns="sequence_id")
    else:
        truth = truth.drop(columns="sequence_id")

    with pytest.raises(ValueError, match="ambiguous sequence_id metadata"):
        score_truth_errors(estimates, truth)


@pytest.mark.parametrize("missing_sequence_from", ["estimates", "truth"])
def test_add_truth_errors_allows_one_sided_sequence_for_unique_flights(
    missing_sequence_from: str,
) -> None:
    estimates, truth = _unique_flight_inputs()
    if missing_sequence_from == "estimates":
        estimates = estimates.drop(columns="sequence_id")
    else:
        truth = truth.drop(columns="sequence_id")

    scored = tracker.add_truth_errors(estimates, truth)

    assert np.allclose(scored["truth_x_m"], [5.0, 105.0])
    assert np.allclose(scored["error_2d_m"], 0.0)
    assert np.allclose(scored["error_3d_m"], 0.0)
