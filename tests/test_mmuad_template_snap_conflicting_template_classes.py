import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_conflicting_duplicate_template_classifications_are_rejected():
    results = pd.DataFrame(
        columns=["Sequence", "Timestamp", "Position", "Classification"]
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [1.0, 1.0],
            "Classification": [1, 3],
        }
    )

    with pytest.raises(ValueError, match="conflicting Classification values"):
        snap_official_results_to_template(results, template)


def test_matching_duplicate_template_classifications_remain_supported():
    results = pd.DataFrame(
        columns=["Sequence", "Timestamp", "Position", "Classification"]
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [1.0, 1.0],
            "Classification": [3, 3],
        }
    )

    snapped, _ = snap_official_results_to_template(results, template)

    assert snapped["Classification"].tolist() == [3, 3]
