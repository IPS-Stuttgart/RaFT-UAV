import numpy as np
import pandas as pd

from raft_uav.io.aerpaw import normalize_truth


def test_normalize_truth_skips_malformed_coordinate_cells():
    truth = pd.DataFrame(
        {
            "timestamp_raw": [
                "2025-10-07 15:42:19",
                "2025-10-07 15:42:20",
                "2025-10-07 15:42:21",
                "2025-10-07 15:42:22",
            ],
            "latitude": [
                "not-a-latitude",
                "35.7274895",
                "35.7274895",
                "35.7274895",
            ],
            "longitude": [
                "-78.696216",
                "not-a-longitude",
                "-78.696216",
                "-78.696216",
            ],
            "altitude_m": [
                "2.717",
                "2.717",
                "not-an-altitude",
                "2.717",
            ],
        }
    )

    normalized, _projector, origin_time = normalize_truth(truth)

    assert origin_time == pd.Timestamp("2025-10-07 15:42:22")
    assert normalized["time_s"].tolist() == [0.0]
    np.testing.assert_allclose(
        normalized[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        np.zeros((1, 3)),
        atol=1e-6,
    )
