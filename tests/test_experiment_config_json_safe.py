import json
from pathlib import Path

import numpy as np

from raft_uav.experiments.config import ExperimentConfig, write_resolved_experiment_config


def test_write_resolved_config_serializes_json_safe_metadata(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="json-safe",
        dataset_root="dataset",
        output_dir="outputs",
        metadata={
            "fold": np.int64(3),
            "source_path": tmp_path / "input.csv",
            "invalid_score": np.nan,
        },
    )
    destination = tmp_path / "resolved.json"

    resolved = write_resolved_experiment_config(
        destination,
        config=config,
        argv=["run-experiment"],
        env_prefixes=(),
        extra={
            "array": np.array([1.0, np.inf]),
            "tuple": (np.float64(2.5), Path("artifact.csv")),
        },
    )

    text = destination.read_text(encoding="utf-8")
    loaded = json.loads(text)

    assert loaded == resolved
    assert "NaN" not in text
    assert "Infinity" not in text
    assert loaded["config"]["metadata"] == {
        "fold": 3,
        "source_path": str(tmp_path / "input.csv"),
        "invalid_score": None,
    }
    assert loaded["extra"] == {
        "array": [1.0, None],
        "tuple": [2.5, "artifact.csv"],
    }
