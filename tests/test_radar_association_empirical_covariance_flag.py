from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.baselines import radar_association
from raft_uav.baselines.radar_association import (
    run_async_cv_baseline_with_radar_association,
)


@pytest.mark.parametrize(
    "value",
    [
        "False",
        "true",
        0,
        1,
        None,
        [],
        {},
        np.array(False),
    ],
)
def test_radar_association_rejects_ambiguous_empirical_covariance_flag(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="paper_compatible_empirical_covariance must be a boolean",
    ):
        run_async_cv_baseline_with_radar_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            association="paper-compatible",
            paper_compatible_empirical_covariance=value,
        )


@pytest.mark.parametrize("value", [False, True, np.bool_(False), np.bool_(True)])
def test_radar_association_normalizes_boolean_empirical_covariance_flag(
    value: object,
) -> None:
    bound = radar_association._RUN_SIGNATURE.bind(
        rf_measurements=[],
        radar=pd.DataFrame(),
        association="paper-compatible",
        paper_compatible_empirical_covariance=value,
    )
    bound.apply_defaults()

    radar_association._validate_radar_association_parameters(bound.arguments)

    normalized = bound.arguments["paper_compatible_empirical_covariance"]
    assert isinstance(normalized, bool)
    assert normalized is bool(value)
