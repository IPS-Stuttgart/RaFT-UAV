from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.experiments.config import (
    ExperimentConfig,
    write_resolved_experiment_config,
)


def test_from_mapping_rejects_environment_keys_that_stringify_identically() -> None:
    with pytest.raises(
        ValueError,
        match=r"environment contains keys 1 and '1'.*normalize to '1'",
    ):
        ExperimentConfig.from_mapping(
            {
                "environment": {
                    1: "integer key",
                    "1": "string key",
                }
            }
        )


def test_writer_rejects_nested_metadata_keys_that_stringify_identically(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        name="collision",
        dataset_root="dataset",
        output_dir="outputs",
        metadata={
            "nested": {
                1: "integer key",
                "1": "string key",
            }
        },
    )
    destination = tmp_path / "resolved.json"

    with pytest.raises(
        ValueError,
        match=r"mapping contains keys 1 and '1'.*normalize to '1'",
    ):
        write_resolved_experiment_config(
            destination,
            config=config,
            argv=["run-experiment"],
            env_prefixes=(),
        )

    assert not destination.exists()


def test_distinct_normalized_mapping_keys_remain_json_safe(tmp_path: Path) -> None:
    config = ExperimentConfig.from_mapping(
        {
            "environment": {1: "integer", "two": 2},
            "metadata": {3: "three"},
        }
    )
    destination = tmp_path / "resolved.json"

    resolved = write_resolved_experiment_config(
        destination,
        config=config,
        argv=["run-experiment"],
        env_prefixes=(),
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == resolved
    assert loaded["config"]["environment"] == {
        "1": "integer",
        "two": "2",
    }
    assert loaded["config"]["metadata"] == {"3": "three"}
