from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_classification_relabel import relabel_track5_classification
from raft_uav.mmuad.track5_classification_relabel import (
    relabel_track5_classification_from_sequence_predictions,
)


def _pose_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0002"],
            "Timestamp": [0.0, 0.0],
            "Position": ["(0,0,1)", "(5,0,2)"],
            "Classification": [0, 3],
        }
    )


def _classification_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0002"],
            "Timestamp": [0.0, 0.0],
            "Position": ["(9,9,9)", "(7,7,7)"],
            "Classification": [1, 2],
        }
    )


@pytest.mark.parametrize("boolean_value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("frame_name", ["pose", "classification"])
def test_classification_relabel_rejects_boolean_official_labels(
    boolean_value: object,
    frame_name: str,
) -> None:
    pose = _pose_rows()
    classification = _classification_rows()
    target = pose if frame_name == "pose" else classification
    target["Classification"] = target["Classification"].astype(object)
    target.loc[0, "Classification"] = boolean_value

    with pytest.raises(ValueError, match="Boolean class labels"):
        relabel_track5_classification(pose, classification)


@pytest.mark.parametrize("boolean_value", [True, False, np.bool_(True), np.bool_(False)])
def test_classification_relabel_rejects_boolean_predicted_classes(
    boolean_value: object,
) -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0002"],
            "predicted_class": pd.Series([boolean_value, 2], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="Boolean class labels"):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )
