from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_utils import (
    load_official_track5_results_frame_from_frame,
)


def _official_results(classification: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA"],
            "Timestamp": [0.0],
            "Position": ["(1,2,3)"],
            "Classification": [classification],
        }
    )


@pytest.mark.parametrize("classification", [1.000001, 2.999999, "1.000001"])
def test_template_snap_rejects_near_integer_classification_ids(
    classification: object,
) -> None:
    with pytest.raises(ValueError, match="Classification values must be integer ids"):
        load_official_track5_results_frame_from_frame(
            _official_results(classification)
        )


@pytest.mark.parametrize("classification", [np.inf, -np.inf, "inf"])
def test_template_snap_rejects_nonfinite_classification_ids(
    classification: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="Classification values must be finite integer ids",
    ):
        load_official_track5_results_frame_from_frame(
            _official_results(classification)
        )


@pytest.mark.parametrize("classification", [1, 1.0, "1.0"])
def test_template_snap_accepts_exact_integer_equivalents(
    classification: object,
) -> None:
    normalized = load_official_track5_results_frame_from_frame(
        _official_results(classification)
    )

    assert normalized["Classification"].tolist() == [1]
