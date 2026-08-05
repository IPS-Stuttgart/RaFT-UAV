from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.stress.perturbations import PerturbationConfig, perturb_radar


def _cyclic_box() -> np.ndarray:
    value = np.empty((), dtype=object)
    value[()] = value
    return value


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        "",
        "   ",
        17,
        b"stress",
        np.nan,
        pd.NA,
        np.ma.masked,
        np.array(["stress"]),
        _cyclic_box(),
    ],
)
def test_perturbation_config_rejects_malformed_names(invalid: object) -> None:
    with pytest.raises(ValueError, match="name must be a non-blank string scalar"):
        PerturbationConfig(name=invalid)


def test_perturbation_config_normalizes_name_for_artifact_provenance() -> None:
    config = PerturbationConfig.from_mapping(
        {
            "name": np.array("  timestamp-jitter  "),
            "timestamp_jitter_std_s": 0.5,
        }
    )

    assert config.name == "timestamp-jitter"

    radar = pd.DataFrame({"time_s": [0.0, 1.0]})
    perturbed = perturb_radar(radar, config)

    assert perturbed["stress_config"].tolist() == [
        "timestamp-jitter",
        "timestamp-jitter",
    ]
