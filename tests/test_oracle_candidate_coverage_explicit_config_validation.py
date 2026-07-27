from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.oracle_candidate_coverage import (
    build_oracle_candidate_coverage_diagnostics,
)


def test_oracle_candidate_coverage_rejects_invalid_config_before_empty_return() -> None:
    for invalid_config in (False, {}, object()):
        with pytest.raises(ValueError, match="TrackletViterbiAssociationConfig"):
            build_oracle_candidate_coverage_diagnostics(
                radar=pd.DataFrame(),
                truth=pd.DataFrame(),
                config=invalid_config,
            )
