from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification
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


def _classification_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(9,9,9)", "(8,8,8)", "(7,7,7)"],
            "Classification": [1, 1, 2],
        }
    )


@pytest.mark.parametrize("bad_label", [1.000001, 3.000001, -1.0e-9])
def test_classification_relabel_rejects_approximately_integer_source_labels(
    bad_label: float,
) -> None:
    source = _classification_rows()
    source.loc[0, "Classification"] = bad_label

    with pytest.raises(ValueError, match="contains non-integer class labels"):
        relabel_track5_classification(_pose_rows(), source)


@pytest.mark.parametrize("bad_label", [1.000001, 3.000001, -1.0e-9])
def test_classification_relabel_rejects_approximately_integer_predictions(
    bad_label: float,
) -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_class": [bad_label, 2.0],
        }
    )

    with pytest.raises(ValueError, match="contains non-integer class labels"):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_classification_relabel_accepts_exact_integer_equivalent_labels() -> None:
    source = _classification_rows().astype({"Classification": float})
    result = relabel_track5_classification(_pose_rows(), source)

    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_class": ["1.0", "2.0"],
        }
    )
    predicted = relabel_track5_classification_from_sequence_predictions(
        _pose_rows(),
        predictions,
    )

    assert result.rows["Classification"].tolist() == [1, 1, 2]
    assert predicted.rows["Classification"].tolist() == [1, 1, 2]
