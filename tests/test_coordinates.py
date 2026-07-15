import numpy as np

from raft_uav.coordinates import LocalENUProjector


def test_transform_many_flattens_broadcast_coordinate_grid() -> None:
    projector = LocalENUProjector(35.7274895, -78.696216, 2.717)
    latitudes = np.array(
        [
            [35.7274895, 35.7275895],
            [35.7276895, 35.7277895],
        ]
    )
    longitudes = np.array(
        [
            [-78.696216, -78.696116],
            [-78.696016, -78.695916],
        ]
    )
    altitude_m = 30.0

    actual = projector.transform_many(latitudes, longitudes, altitude_m)
    expected = np.vstack(
        [
            projector.transform(latitude, longitude, altitude_m)
            for latitude, longitude in zip(
                latitudes.ravel(),
                longitudes.ravel(),
                strict=True,
            )
        ]
    )

    assert actual.shape == (latitudes.size, 3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-9)
