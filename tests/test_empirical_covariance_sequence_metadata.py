from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.calibration.empirical_covariance import aligned_residuals


def _rf(*, include_sequence_id: bool) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0],
        "east_m": [101.0],
        "north_m": [99.0],
    }
    if include_sequence_id:
        data["sequence_id"] = ["seq_b"]
    return pd.DataFrame(data)


def _truth(*, include_sequence_id: bool) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0, 0.0],
        "east_m": [0.0, 100.0],
        "north_m": [0.0, 100.0],
    }
    if include_sequence_id:
        data["sequence_id"] = ["seq_a", "seq_b"]
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("frame_has_sequence_id", "truth_has_sequence_id"),
    [(True, False), (False, True)],
)
def test_empirical_covariance_rejects_one_sided_sequence_metadata(
    frame_has_sequence_id: bool,
    truth_has_sequence_id: bool,
) -> None:
    with pytest.raises(
        ValueError,
        match="frame and truth must either both contain sequence_id or both omit it",
    ):
        aligned_residuals(
            _rf(include_sequence_id=frame_has_sequence_id),
            _truth(include_sequence_id=truth_has_sequence_id),
            source="rf",
            max_time_delta_s=0.25,
        )


def test_empirical_covariance_keeps_sequence_free_alignment() -> None:
    residuals = aligned_residuals(
        _rf(include_sequence_id=False),
        _truth(include_sequence_id=False).iloc[[1]],
        source="rf",
        max_time_delta_s=0.25,
    )

    assert residuals.tolist() == [[1.0, -1.0]]


@pytest.mark.parametrize(
    ("frame_has_flight_id", "truth_has_flight_id"),
    [(True, False), (False, True)],
)
def test_empirical_covariance_rejects_one_sided_flight_metadata(
    frame_has_flight_id: bool,
    truth_has_flight_id: bool,
) -> None:
    frame = _rf(include_sequence_id=False)
    truth = _truth(include_sequence_id=False)
    if frame_has_flight_id:
        frame["flight_id"] = ["flight_b"]
    if truth_has_flight_id:
        truth["flight_id"] = ["flight_a", "flight_b"]

    with pytest.raises(
        ValueError,
        match="frame and truth must either both contain flight_id or both omit it",
    ):
        aligned_residuals(
            frame,
            truth,
            source="rf",
            max_time_delta_s=0.25,
        )


def test_empirical_covariance_scopes_alignment_by_flight_id() -> None:
    frame = _rf(include_sequence_id=False).assign(flight_id="flight_b")
    truth = _truth(include_sequence_id=False).assign(
        flight_id=["flight_a", "flight_b"]
    )

    residuals = aligned_residuals(
        frame,
        truth,
        source="rf",
        max_time_delta_s=0.25,
    )

    assert residuals.tolist() == [[1.0, -1.0]]


def test_empirical_covariance_uses_joint_sequence_and_flight_scope() -> None:
    frame = _rf(include_sequence_id=True).assign(flight_id="flight_b")
    frame["sequence_id"] = "shared"
    truth = _truth(include_sequence_id=True).assign(
        sequence_id=["shared", "shared"],
        flight_id=["flight_a", "flight_b"],
    )

    residuals = aligned_residuals(
        frame,
        truth,
        source="rf",
        max_time_delta_s=0.25,
    )

    assert residuals.tolist() == [[1.0, -1.0]]
