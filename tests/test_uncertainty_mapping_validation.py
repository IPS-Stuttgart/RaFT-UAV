from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import pytest

from raft_uav.uncertainty import fit_heteroscedastic_uncertainty_model


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [0.0, 0.0],
            "north_m": [0.0, 0.0],
        }
    )
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [1.0, 2.0],
            "north_m": [1.0, 2.0],
        }
    )
    return truth, rf


def _fit(**kwargs):
    truth, rf = _frames()
    return fit_heteroscedastic_uncertainty_model(
        rf=rf,
        radar=None,
        truth=truth,
        **kwargs,
    )


@pytest.mark.parametrize("field", ["min_std_m", "max_std_m", "metadata"])
@pytest.mark.parametrize(
    "invalid_mapping",
    [
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param([("rf", {})], id="pair-list"),
    ],
)
def test_fit_rejects_non_mapping_configuration(
    field: str,
    invalid_mapping: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a mapping or None"):
        _fit(**{field: invalid_mapping})


def test_fit_accepts_empty_mapping_configuration() -> None:
    model = _fit(min_std_m={}, max_std_m={}, metadata={})

    assert isinstance(model.metadata, Mapping)
    assert model.metadata["ridge_lambda"] == 1.0
