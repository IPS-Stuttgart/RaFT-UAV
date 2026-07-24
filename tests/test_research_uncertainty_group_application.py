from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.uncertainty import (
    ConformalRadius,
    apply_group_conformal_radius,
)


def test_unseen_conformal_group_does_not_borrow_arbitrary_radius() -> None:
    frame = pd.DataFrame({"phase": ["day", "unseen"]})
    radii = {
        "day": ConformalRadius(radius_m=5.0, alpha=0.1, sample_count=20),
        "night": ConformalRadius(radius_m=50.0, alpha=0.1, sample_count=20),
    }

    applied = apply_group_conformal_radius(frame, radii)
    reversed_applied = apply_group_conformal_radius(
        frame,
        dict(reversed(list(radii.items()))),
    )

    assert applied.loc[0, "conformal_radius_m"] == 5.0
    assert np.isnan(applied.loc[1, "conformal_radius_m"])
    assert np.isnan(reversed_applied.loc[1, "conformal_radius_m"])
