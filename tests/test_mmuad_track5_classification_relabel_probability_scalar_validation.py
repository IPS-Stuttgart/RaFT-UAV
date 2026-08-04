from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import (
    relabel_track5_classification_from_sequence_predictions,
)


def _pose_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,1)", "(1,0,1)", "(5,0,2)"],
            "Classification": [0, 0, 3],
        }
    )


def _boxed(value: object) -> np.ndarray:
    inner = np.empty((), dtype=object)
    inner[()] = value
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.complex128(0.75 + 2.0j),
        np.array(True, dtype=object),
        _boxed(np.complex64(0.5 + 0.0j)),
    ],
)
def test_sequence_probability_relabel_rejects_lossy_probability_scalars(
    invalid_value: object,
) -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [invalid_value, 0.25],
            "predicted_probability_1": [0.5, 0.75],
        }
    )

    with pytest.raises(
        ValueError,
        match="non-finite, non-numeric, or negative values",
    ):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_sequence_probability_relabel_accepts_boxed_real_scalars() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_probability_0": [_boxed(2.0), np.array(7.0)],
            "predicted_probability_1": [np.array(8.0), _boxed(3.0)],
        }
    )

    result = relabel_track5_classification_from_sequence_predictions(
        _pose_rows(),
        predictions,
    )

    assert result.rows["Classification"].tolist() == [1, 1, 0]
    assert result.diagnostics["source_classification_probability"].tolist() == pytest.approx(
        [0.8, 0.8, 0.7]
    )
