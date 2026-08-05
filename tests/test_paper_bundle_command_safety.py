from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.research.paper_bundle import (
    ReproducibilityCommand,
    write_reproducibility_bundle,
)


@pytest.mark.parametrize(
    "name",
    [
        "",
        " leading-space",
        "trailing-space ",
        ".",
        "..",
        "../outside",
        "/tmp/outside",
        "nested/output",
        "line\nbreak",
    ],
)
def test_reproducibility_bundle_rejects_unsafe_command_names(
    tmp_path: Path,
    name: str,
) -> None:
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="command names"):
        write_reproducibility_bundle(
            output_dir,
            commands=[ReproducibilityCommand(name=name, command=["python", "-V"])],
            config={},
        )

    assert not output_dir.exists()


def test_reproducibility_bundle_rejects_duplicate_log_names(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="unique"):
        write_reproducibility_bundle(
            output_dir,
            commands=[
                ReproducibilityCommand(name="evaluate", command=["python", "a.py"]),
                ReproducibilityCommand(name="evaluate", command=["python", "b.py"]),
            ],
            config={},
        )

    assert not output_dir.exists()


def test_reproducibility_bundle_shell_quotes_readme_commands(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    command = ReproducibilityCommand(
        name="evaluate",
        command=["python", "script name.py", "--label", "two words", "$HOME"],
    )

    manifest = write_reproducibility_bundle(
        output_dir,
        commands=[command],
        config={"dataset": "example"},
    )

    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "python 'script name.py' --label 'two words' '$HOME'" in readme
    assert manifest["commands"] == [
        {
            "name": "evaluate",
            "command": command.command,
            "description": "",
        }
    ]
