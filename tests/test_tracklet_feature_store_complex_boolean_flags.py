from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.tracklet_feature_store import (
    build_counterfactual_association_dashboard,
    summarize_counterfactual_regret,
)


def _nested_complex_scalar(value: complex) -> np.ndarray:
    inner = np.asarray(value)
    outer = np.empty((), dtype=object)
    outer[()] = inner
    return outer


_COMPLEX_BOOLEAN_VALUES = (
    1.0 + 0.0j,
    0.0 + 0.0j,
    np.complex128(1.0 + 0.0j),
    np.asarray(0.0 + 0.0j),
    _nested_complex_scalar(1.0 + 0.0j),
)


@pytest.mark.parametrize("value", _COMPLEX_BOOLEAN_VALUES)
def test_dashboard_rejects_complex_selection_flags(value: object) -> None:
    features = pd.DataFrame(
        {
            "frame_key_type": ["frame_index"],
            "frame_key": ["7"],
            "time_s": [1.0],
            "oracle_error_m": [1.0],
            "oracle_rank_in_frame": [1.0],
            "chosen_by_selected_radar": pd.Series(
                [value],
                index=[73],
                dtype=object,
            ),
        },
        index=[73],
    )

    with pytest.raises(
        ValueError,
        match=r"chosen_by_selected_radar contains invalid Boolean values at rows \[73\]",
    ):
        build_counterfactual_association_dashboard(features)


@pytest.mark.parametrize("value", _COMPLEX_BOOLEAN_VALUES)
@pytest.mark.parametrize("column", ["truth_available", "selected_present"])
def test_regret_summary_rejects_complex_boolean_flags(
    value: object,
    column: str,
) -> None:
    regret = pd.DataFrame(
        {
            "truth_available": [True],
            "selected_present": [True],
            "selection_regret_m": [1.0],
            "category": ["wrong_candidate_selected"],
        },
        index=[73],
    )
    regret[column] = pd.Series([value], index=regret.index, dtype=object)

    with pytest.raises(
        ValueError,
        match=rf"{column} contains invalid Boolean values at rows \[73\]",
    ):
        summarize_counterfactual_regret(regret)
