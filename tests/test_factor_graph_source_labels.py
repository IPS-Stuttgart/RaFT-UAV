from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    _row_position_std,
)


@pytest.mark.parametrize("source", ["RF", " rf ", "Rf"])
def test_factor_graph_normalizes_rf_source_labels(source: str) -> None:
    config = LeastSquaresSmoothingConfig(measurement_std_m=25.0, rf_std_m=50.0)

    np.testing.assert_allclose(
        _row_position_std(pd.Series({"source": source}), config),
        [50.0, 50.0, 50.0],
    )
