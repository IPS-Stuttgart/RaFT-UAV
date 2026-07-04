from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.mmuad.track5_estimate_ensemble_apply_weights import load_ensemble_weight_config


def test_load_ensemble_weight_config_rejects_duplicate_normalized_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.json"
    path.write_text(
        json.dumps(
            {
                "weights": {
                    "candidate/a": 0.75,
                    "candidate_a": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate normalized weight label"):
        load_ensemble_weight_config(path)
