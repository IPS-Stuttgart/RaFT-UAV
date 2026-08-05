from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.experiments.config import ExperimentConfig, write_resolved_experiment_config


def test_write_resolved_config_normalizes_pandas_missing_scalars(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="pandas-missing",
        dataset_root="dataset",
        output_dir="outputs",
        metadata={
            "optional_label": pd.NA,
            "optional_timestamp": pd.NaT,
        },
    )
    destination = tmp_path / "resolved.json"

    resolved = write_resolved_experiment_config(
        destination,
        config=config,
        argv=["run-experiment"],
        env_prefixes=(),
        extra={
            "object_array": np.array([pd.NA, np.int64(4)], dtype=object),
        },
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == resolved
    assert loaded["config"]["metadata"] == {
        "optional_label": None,
        "optional_timestamp": None,
    }
    assert loaded["extra"]["object_array"] == [None, 4]
