from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.train_selected_config import (
    build_train_selected_config,
    load_train_selected_config,
    validate_train_selected_config,
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_translation_alpha", -0.01),
        ("source_translation_alpha", 1.01),
        ("smoothing_blend", -0.01),
        ("smoothing_blend", 1.01),
    ],
)
def test_train_selected_config_rejects_out_of_range_unit_interval_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be in \[0, 1\]"):
        validate_train_selected_config({field: value})


def test_train_selected_config_keeps_unit_interval_boundaries() -> None:
    config = validate_train_selected_config(
        {
            "source_translation_alpha": 0.0,
            "smoothing_blend": 1.0,
        }
    )

    assert config["source_translation_alpha"] == pytest.approx(0.0)
    assert config["smoothing_blend"] == pytest.approx(1.0)


def test_loaded_train_selected_config_rejects_alpha_instead_of_clipping(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "selected.json"
    config_path.write_text(
        json.dumps({"config": {"source_translation_alpha": 1.5}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"source_translation_alpha must be in \[0, 1\]"):
        load_train_selected_config(config_path)


def test_train_selected_config_rejects_out_of_range_selected_summary(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "source_calibration_summary.csv"
    pd.DataFrame(
        {
            "source_calibration_mode": ["source-translation"],
            "source_translation_alpha": [-0.25],
            "after_mean_m": [2.0],
        }
    ).to_csv(summary_path, index=False)

    with pytest.raises(ValueError, match=r"source_translation_alpha must be in \[0, 1\]"):
        build_train_selected_config(source_calibration_summary_csv=summary_path)
