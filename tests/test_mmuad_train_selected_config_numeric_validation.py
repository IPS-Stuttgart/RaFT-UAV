from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.train_selected_config import (
    load_train_selected_config,
    validate_train_selected_config,
)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        [0.5],
        np.array([0.5]),
        1.0 + 0.0j,
        np.complex128(1.0 + 0.0j),
        np.ma.masked,
    ],
)
def test_train_selected_config_rejects_non_real_scalar_numeric_controls(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="expected finite float"):
        validate_train_selected_config({"smoothing_blend": value})


def test_train_selected_config_rejects_json_boolean_numeric_control(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "selected.json"
    config_path.write_text(
        json.dumps({"config": {"image_nonimage_fusion_weight": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected finite float"):
        load_train_selected_config(config_path)


def test_train_selected_config_keeps_valid_scalar_numeric_controls() -> None:
    config = validate_train_selected_config(
        {
            "source_translation_alpha": np.float64(0.25),
            "smoothing_blend": "0.5",
            "image_nonimage_fusion_weight": 0,
        }
    )

    assert config["source_translation_alpha"] == pytest.approx(0.25)
    assert config["smoothing_blend"] == pytest.approx(0.5)
    assert config["image_nonimage_fusion_weight"] == pytest.approx(0.0)
