from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import (
    apply_ensemble_weight_config,
)


def test_apply_ensemble_weight_config_rejects_unused_weight_labels() -> None:
    with pytest.raises(ValueError, match="labels not present"):
        apply_ensemble_weight_config(
            ["alpha=/tmp/alpha.csv"],
            {"weights": {"alpha": 1.0, "typo": 0.5}},
        )


def test_apply_ensemble_weight_config_allows_default_for_missing_inputs() -> None:
    inputs = apply_ensemble_weight_config(
        ["alpha=/tmp/alpha.csv", "beta=/tmp/beta.csv"],
        {"weights": {"alpha": 1.0}},
        missing_weight_policy="default",
        default_missing_weight=0.25,
    )

    assert [item.label for item in inputs] == ["alpha", "beta"]
    assert [item.path for item in inputs] == [Path("/tmp/alpha.csv"), Path("/tmp/beta.csv")]
    assert [item.weight for item in inputs] == [1.0, 0.25]
