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
