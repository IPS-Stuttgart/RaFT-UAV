from __future__ import annotations

import os

import pytest

from raft_uav import bias_model_cli, tracklet_viterbi_cli
from raft_uav.calibration import bias_runtime
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV


def test_bias_wrapper_installs_environment_configured_model(monkeypatch) -> None:
    monkeypatch.setenv(BIAS_MODEL_ENV, "environment-model.json")
    installed_paths: list[str | None] = []

    monkeypatch.setattr(
        bias_runtime,
        "install",
        lambda: installed_paths.append(os.environ.get(BIAS_MODEL_ENV)),
    )
    monkeypatch.setattr(bias_model_cli, "_refresh_cli_normalizers", lambda: None)
    monkeypatch.setattr(tracklet_viterbi_cli, "main", lambda argv: 7)

    status = bias_model_cli.main(["run-baseline", "dataset"])

    assert status == 7
    assert installed_paths == ["environment-model.json"]


@pytest.mark.parametrize("previous", [None, "existing-model.json"])
def test_explicit_bias_model_is_scoped_to_one_invocation(
    monkeypatch,
    previous: str | None,
) -> None:
    if previous is None:
        monkeypatch.delenv(BIAS_MODEL_ENV, raising=False)
    else:
        monkeypatch.setenv(BIAS_MODEL_ENV, previous)
    active_paths: list[str | None] = []

    monkeypatch.setattr(bias_runtime, "install", lambda: None)
    monkeypatch.setattr(bias_model_cli, "_refresh_cli_normalizers", lambda: None)

    def fake_main(argv: list[str]) -> int:
        active_paths.append(os.environ.get(BIAS_MODEL_ENV))
        assert argv == ["run-baseline", "dataset"]
        return 9

    monkeypatch.setattr(tracklet_viterbi_cli, "main", fake_main)

    status = bias_model_cli.main(
        ["--bias-model", "one-shot-model.json", "run-baseline", "dataset"]
    )

    assert status == 9
    assert active_paths == ["one-shot-model.json"]
    assert os.environ.get(BIAS_MODEL_ENV) == previous
