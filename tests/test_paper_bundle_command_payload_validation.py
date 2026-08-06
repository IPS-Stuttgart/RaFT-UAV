from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.research.paper_bundle import (
    ReproducibilityCommand,
    _validated_commands,
    write_reproducibility_bundle,
)


@pytest.mark.parametrize(
    ("argv", "error_type", "message"),
    [
        ([], ValueError, "argv must not be empty"),
        ("python -V", TypeError, "argv must be a list of strings"),
        (["python", None], TypeError, "argv entries must be strings"),
        ([""], ValueError, "executable must be non-empty"),
        (["   "], ValueError, "executable must be non-empty"),
        (["python", "bad\x00argument"], ValueError, "must not contain NUL"),
    ],
)
def test_reproducibility_bundle_rejects_invalid_command_argv_without_side_effects(
    tmp_path: Path,
    argv: object,
    error_type: type[Exception],
    message: str,
) -> None:
    output_dir = tmp_path / "bundle"

    with pytest.raises(error_type, match=message):
        write_reproducibility_bundle(
            output_dir,
            commands=[ReproducibilityCommand(name="evaluate", command=argv)],
            config={},
        )

    assert not output_dir.exists()


def test_reproducibility_bundle_rejects_non_string_description_without_side_effects(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "bundle"

    with pytest.raises(TypeError, match="descriptions must be strings"):
        write_reproducibility_bundle(
            output_dir,
            commands=[
                ReproducibilityCommand(
                    name="evaluate",
                    command=["python", "-V"],
                    description=123,
                )
            ],
            config={},
        )

    assert not output_dir.exists()


def test_validated_reproducibility_commands_snapshot_mutable_argv() -> None:
    argv = ["python", "-V"]
    command = ReproducibilityCommand(name="evaluate", command=argv)

    validated = _validated_commands([command])
    argv.append("--unexpected")

    assert validated[0].command == ["python", "-V"]
