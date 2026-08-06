from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.experiments.config import filtered_environment, write_resolved_experiment_config


def test_scalar_environment_prefix_is_not_split_into_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAFT_UAV_VISIBLE_SETTING", "enabled")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-captured")

    captured = filtered_environment("RAFT_UAV_")

    assert captured["RAFT_UAV_VISIBLE_SETTING"] == "enabled"
    assert "AWS_SECRET_ACCESS_KEY" not in captured


def test_resolved_config_uses_scalar_environment_prefix_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAFT_UAV_VISIBLE_SETTING", "enabled")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-captured")

    resolved = write_resolved_experiment_config(
        tmp_path / "resolved.json",
        argv=["raft-uav"],
        env_prefixes="RAFT_UAV_",
    )

    assert resolved["environment"]["RAFT_UAV_VISIBLE_SETTING"] == "enabled"
    assert "AWS_SECRET_ACCESS_KEY" not in resolved["environment"]


@pytest.mark.parametrize("prefixes", ["", ("",), ("RAFT_UAV_", 7)])
def test_environment_prefixes_reject_unsafe_values(prefixes: object) -> None:
    with pytest.raises(ValueError, match="environment prefixes"):
        filtered_environment(prefixes)  # type: ignore[arg-type]
