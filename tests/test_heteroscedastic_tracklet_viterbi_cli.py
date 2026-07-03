from __future__ import annotations

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
