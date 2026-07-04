from __future__ import annotations

import pytest

from raft_uav import heteroscedastic_tracklet_viterbi_cli as cli
from raft_uav.heteroscedastic_tracklet_viterbi_cli import (
    _ensure_tracklet_viterbi_association,
)


def test_tracklet_association_is_appended_when_missing() -> None:
    assert _ensure_tracklet_viterbi_association(["run-baseline", "DATA"]) == [
        "run-baseline",
        "DATA",
        "--radar-association",
        "tracklet-viterbi",
    ]


def test_explicit_tracklet_association_is_preserved() -> None:
    argv = ["run-baseline", "DATA", "--radar-association=prediction-nis"]

    assert _ensure_tracklet_viterbi_association(argv) is argv


def test_tracklet_association_is_inserted_before_option_sentinel() -> None:
    assert _ensure_tracklet_viterbi_association(
        ["run-baseline", "DATA", "--", "--not-a-wrapper-option"]
    ) == [
        "run-baseline",
        "DATA",
        "--radar-association",
        "tracklet-viterbi",
        "--",
        "--not-a-wrapper-option",
    ]


def test_help_is_forwarded_without_uncertainty_model(monkeypatch: pytest.MonkeyPatch) -> None:
    forwarded: list[list[str]] = []

    def fake_tracklet_main(argv: list[str] | None = None) -> int:
        forwarded.append(list(argv or []))
        return 0

    monkeypatch.setattr(cli.tracklet_viterbi_cli, "main", fake_tracklet_main)

    assert cli.main(["run-baseline", "--help"]) == 0
    assert forwarded == [["run-baseline", "--help", "--radar-association", "tracklet-viterbi"]]
