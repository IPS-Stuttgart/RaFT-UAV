from __future__ import annotations

import pytest

import raft_uav.mmuad.sequence_alignment_audit as sequence_alignment_audit


def test_sequence_alignment_cli_defaults_to_all_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 17

    monkeypatch.setattr(sequence_alignment_audit, "_ORIGINAL_MAIN", fake_main)
    status = sequence_alignment_audit.main(
        [
            "dataset",
            "--truth-file",
            "truth.csv",
            "--output-dir",
            "out",
        ]
    )

    assert status == 17
    assert captured["argv"][-2:] == ["--sequence-glob", "*"]
    assert captured["argv"].count("--sequence-glob") == 1


@pytest.mark.parametrize(
    "explicit_glob",
    [
        ["--sequence-glob", "seq0003"],
        ["--sequence-glob=seq0004"],
    ],
)
def test_sequence_alignment_cli_preserves_explicit_sequence_glob(
    monkeypatch: pytest.MonkeyPatch,
    explicit_glob: list[str],
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(sequence_alignment_audit, "_ORIGINAL_MAIN", fake_main)
    arguments = [
        "dataset",
        "--truth-file",
        "truth.csv",
        "--output-dir",
        "out",
        *explicit_glob,
    ]

    status = sequence_alignment_audit.main(arguments)

    assert status == 0
    assert captured["argv"] == arguments
