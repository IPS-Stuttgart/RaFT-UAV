from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.uncertainty import feature_matrix


def test_rf_valid_sensor_fraction_is_clipped_to_fraction_domain() -> None:
    frame = pd.DataFrame(
        {
            "CEP": [0.0, 0.0, 0.0],
            "RHO": [0.0, 0.0, 0.0],
            "ValidSensors": [-2, 12, 3],
            "TotalSensors": [4, 4, 0],
        }
    )

    features = feature_matrix(frame, "rf", ("valid_sensor_fraction",))

    np.testing.assert_allclose(features[:, 0], [0.0, 1.0, 1.0])
