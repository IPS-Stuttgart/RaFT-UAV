from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def _results(classification: object) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [1.0],
            "Position": ["(1,2,3)"],
            "Classification": [classification],
        }
    )


_TEMPLATE = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [1.0]})


@pytest.mark.parametrize("classification", [1.000001, "2.000001"])
def test_template_snap_rejects_near_integer_classifications(
    classification: object,
) -> None:
    with pytest.raises(ValueError, match="must be integer ids"):
        snap_official_results_to_template(_results(classification), _TEMPLATE)


@pytest.mark.parametrize("classification", [1.0, "2.0"])
def test_template_snap_accepts_exact_integral_classifications(
    classification: object,
) -> None:
    snapped, _ = snap_official_results_to_template(_results(classification), _TEMPLATE)

    assert int(snapped.loc[0, "Classification"]) == int(float(classification))
