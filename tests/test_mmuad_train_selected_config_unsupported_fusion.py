from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.mmuad.train_selected_config import (
    load_train_selected_config,
    validate_train_selected_config,
)


def test_train_selected_config_accepts_zero_image_fusion_weight() -> None:
    config = validate_train_selected_config({"image_nonimage_fusion_weight": 0.0})

    assert config["image_nonimage_fusion_weight"] == 0.0


def test_train_selected_config_rejects_ignored_image_fusion_weight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selected.json"
    path.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-train-selected-config-v1",
                "config": {"image_nonimage_fusion_weight": 0.75},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not consume image-fusion outputs"):
        load_train_selected_config(path)
