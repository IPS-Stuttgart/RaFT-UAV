from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.baselines.tracklet_viterbi import (
    run_async_cv_baseline_with_tracklet_viterbi_association,
)


def test_tracklet_viterbi_rejects_falsy_explicit_config_before_empty_return() -> None:
    with pytest.raises(ValueError, match="TrackletViterbiAssociationConfig"):
        run_async_cv_baseline_with_tracklet_viterbi_association(
            rf_measurements=[],
            radar=pd.DataFrame(),
            config=False,
        )
