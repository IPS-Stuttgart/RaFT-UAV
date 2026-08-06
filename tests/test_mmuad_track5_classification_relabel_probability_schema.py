from __future__ import annotations

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


def test_classification_relabel_rejects_competing_probability_aliases() -> None:
    predictions = pd.DataFrame(
        {
            "heldout_sequence": ["seq0001", "seq0002"],
            0: [0.90, 0.90],
            "predicted_probability_0": [0.00, 0.00],
            "predicted_probability_1": [0.10, 0.10],
        }
    )

    with pytest.raises(ValueError, match="multiple probability columns.*class 0"):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )


def test_classification_relabel_rejects_ignored_out_of_domain_probability() -> None:
    predictions = pd.DataFrame(
        {
            "heldout_sequence": ["seq0001", "seq0002"],
            "predicted_probability_0": [0.10, 0.10],
            "predicted_probability_1": [0.20, 0.20],
            "predicted_probability_4": [0.70, 0.70],
        }
    )

    with pytest.raises(ValueError, match="outside official classes.*probability_4"):
        relabel_track5_classification_from_sequence_predictions(
            _pose_rows(),
            predictions,
        )
