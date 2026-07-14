from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf, -0.25])
def test_uncertainty_ensemble_rejects_invalid_programmatic_weights(
    tmp_path: Path,
    weight: float,
) -> None:
    empty_template = pd.DataFrame(columns=["Sequence", "Timestamp"])

    with pytest.raises(ValueError, match="estimate weight must be finite and non-negative"):
        build_track5_uncertainty_ensemble(
            [EstimateInput("invalid", tmp_path / "not-read.csv", weight)],
            template=empty_template,
        )
