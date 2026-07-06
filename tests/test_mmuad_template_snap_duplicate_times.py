import pandas as pd

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_duplicate_time_nearest_label_matches_last_position_row():
    results = pd.DataFrame({
        "Sequence": ["seq001", "seq001"],
        "Timestamp": [1.0, 1.0],
        "Position": ["(1,1,1)", "(2,2,2)"],
        "Classification": [1, 3],
    })
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [1.0]})

    snapped, _ = snap_official_results_to_template(
        results,
        template,
        resample_method="nearest",
        classification_policy="nearest",
    )

    assert snapped.loc[0, "Position"] == "(2,2,2)"
    assert int(snapped.loc[0, "Classification"]) == 3
