from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_utils import load_official_track5_results_frame_from_frame


def _official_rows(classification: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [classification],
        }
    )


def test_template_snap_rejects_near_integer_classification_labels() -> None:
    with pytest.raises(ValueError, match="integer ids"):
        load_official_track5_results_frame_from_frame(_official_rows("1.000001"))


def test_template_snap_accepts_exact_integer_like_classification_labels() -> None:
    rows = load_official_track5_results_frame_from_frame(_official_rows("1.0"))

    assert rows["Classification"].tolist() == [1]
