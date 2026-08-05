"""Helpers for reproducible paper-result bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReproducibilityCommand:
    """One command included in a reproducibility bundle."""

    name: str
    command: list[str]
    description: str = ""


def git_sha(repo_root: Path | None = None) -> str:
    """Return the current git SHA, or ``unknown`` outside a git checkout."""

    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _validated_commands(
    commands: list[ReproducibilityCommand],
) -> list[ReproducibilityCommand]:
    """Validate and snapshot commands before writing bundle artifacts."""

    validated: list[ReproducibilityCommand] = []
    seen_log_names: set[str] = set()
    for command in list(commands):
        if not isinstance(command, ReproducibilityCommand):
            raise TypeError("commands must contain ReproducibilityCommand values")
        name = command.name
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or name in {".", ".."}
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ValueError(
                "reproducibility command names must be non-empty, trimmed log-file stems"
            )
        log_name = f"{name}.log"
        if Path(log_name).name != log_name:
            raise ValueError(
                "reproducibility command names must not contain path separators"
            )
        normalized_log_name = os.path.normcase(log_name)
        if normalized_log_name in seen_log_names:
            raise ValueError("reproducibility command names must be unique")
        seen_log_names.add(normalized_log_name)

        argv = command.command
        if not isinstance(argv, list):
            raise TypeError("reproducibility command argv must be a list of strings")
        if not argv:
            raise ValueError("reproducibility command argv must not be empty")
        if any(not isinstance(argument, str) for argument in argv):
            raise TypeError("reproducibility command argv entries must be strings")
        if not argv[0].strip():
            raise ValueError("reproducibility command executable must be non-empty")
        if any("\x00" in argument for argument in argv):
            raise ValueError("reproducibility command argv must not contain NUL bytes")
        if not isinstance(command.description, str):
            raise TypeError("reproducibility command descriptions must be strings")

        validated.append(
            ReproducibilityCommand(
                name=name,
                command=list(argv),
                description=command.description,
            )
        )
    return validated


def write_reproducibility_bundle(
    output_dir: Path,
    *,
    commands: list[ReproducibilityCommand],
    config: dict[str, Any],
    dry_run: bool = True,
) -> dict[str, Any]:
    """Write manifest, README, and optional command outputs for paper results."""

    validated_commands = _validated_commands(commands)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": git_sha(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "environment_overrides": {
            key: value for key, value in os.environ.items() if key.startswith("RAFT_UAV_")
        },
        "config": config,
        "commands": [asdict(command) for command in validated_commands],
        "dry_run": bool(dry_run),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    readme_lines = [
        "# RaFT-UAV reproducibility bundle",
        "",
        f"Git SHA: `{manifest['git_sha']}`",
        "",
        "## Commands",
        "",
    ]
    for command in validated_commands:
        readme_lines.append(f"### {command.name}")
        if command.description:
            readme_lines.append(command.description)
        readme_lines.append("")
        readme_lines.append("```bash")
        readme_lines.append(shlex.join(command.command))
        readme_lines.append("```")
        readme_lines.append("")
        if not dry_run:
            log_path = output_dir / f"{command.name}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command.command,
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return manifest
