from pathlib import Path

import numpy as np
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import _select_trim_fraction
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import apply_ensemble_weight_config
from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import load_ensemble_weight_config


def test_apply_weight_config_rejects_negative_in_memory_weight() -> None:
    config = {"weights": {"a": -1.0}}

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_ensemble_weight_config(["a=estimate.csv"], config)


def test_apply_weight_config_rejects_nonfinite_default_missing_weight() -> None:
    config = {"weights": {"a": 1.0}}

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_ensemble_weight_config(
            ["b=estimate.csv"],
            config,
            missing_weight_policy="default",
            default_missing_weight=np.inf,
        )


def test_apply_weight_config_normalizes_valid_in_memory_weights() -> None:
    config = {"weights": {"model a": "0.25", "model/b": 0.75}}

    result = apply_ensemble_weight_config(
        [
            EstimateInput("model a", Path("a.csv"), 1.0),
            EstimateInput("model/b", Path("b.csv"), 1.0),
        ],
        config,
    )

    assert [(item.label, item.weight) for item in result] == [("model_a", 0.25), ("model_b", 0.75)]


def test_load_weight_config_rejects_non_numeric_weight(tmp_path: Path) -> None:
    path = tmp_path / "weights.json"
    path.write_text('{"weights": {"a": "bad"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="finite and non-negative"):
        load_ensemble_weight_config(path)


def test_select_trim_fraction_treats_null_config_value_as_default() -> None:
    assert _select_trim_fraction(None, {"trim_fraction": None}) == pytest.approx(0.2)


def test_select_trim_fraction_prefers_cli_override_over_null_config_value() -> None:
    assert _select_trim_fraction(0.1, {"trim_fraction": None}) == pytest.approx(0.1)


def test_select_trim_fraction_rejects_out_of_range_config_value() -> None:
    with pytest.raises(ValueError, match="trim_fraction"):
        _select_trim_fraction(None, {"trim_fraction": 0.5})
