from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [1],
        }
    )


def _boxed(value: object) -> np.ndarray:
    boxed = np.empty((), dtype=object)
    boxed[()] = value
    return boxed


@pytest.mark.parametrize(
    "tolerance",
    [
        pytest.param(_boxed(_boxed(np.bool_(True))), id="nested-boolean"),
        pytest.param(
            _boxed(_boxed(np.complex128(0.25 + 1.0j))),
            id="nested-complex",
        ),
        pytest.param(_boxed(_boxed(np.asarray([0.25]))), id="nested-vector"),
    ],
)
def test_nearest_time_relabel_rejects_nested_invalid_delta_gate(
    tolerance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_nearest_time_delta_s must be a finite non-negative number",
    ):
        relabel_track5_classification(
            _rows(),
            _rows(),
            mode="by-nearest-time",
            max_nearest_time_delta_s=tolerance,
        )


def test_nearest_time_relabel_rejects_cyclic_delta_gate() -> None:
    tolerance = np.empty((), dtype=object)
    tolerance[()] = tolerance

    with pytest.raises(
        ValueError,
        match="max_nearest_time_delta_s must be a finite non-negative number",
    ):
        relabel_track5_classification(
            _rows(),
            _rows(),
            mode="by-nearest-time",
            max_nearest_time_delta_s=tolerance,
        )


def test_nearest_time_relabel_accepts_nested_real_delta_gate() -> None:
    result = relabel_track5_classification(
        _rows(),
        _rows(),
        mode="by-nearest-time",
        max_nearest_time_delta_s=_boxed(_boxed(np.float64(0.0))),
    )

    assert result.rows["Classification"].tolist() == [1]
    assert result.manifest["max_nearest_time_delta_s"] == 0.0
