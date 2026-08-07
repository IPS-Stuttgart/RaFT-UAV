from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.diagnostics.nis_reliability import nis_reliability_summary


def test_nis_reliability_accepted_only_requires_acceptance_metadata() -> None:
    frame = pd.DataFrame(
        {
            "source": ["radar", "radar"],
            "measurement_dim": [3, 3],
            "nis": [1.0, 2.0],
        }
    )

    with pytest.raises(KeyError, match="accepted_only=True.*accepted"):
        nis_reliability_summary(frame, accepted_only=True)
