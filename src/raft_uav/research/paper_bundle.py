"""Helpers for reproducible paper-result bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import platform
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


def write_reproducibility_bundle(
    output_dir: Path,
    *,
    commands: list[ReproducibilityCommand],
    config: dict[str, Any],
    dry_run: bool = True,
) -> dict[str, Any]:
    """Write manifest, README, and optional command outputs for paper results."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": git_sha(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "environment_overrides": {k: v for k, v in os.environ.items() if k.startswith("RAFT_UAV_")},
        "config": config,
        "commands": [asdict(command) for command in commands],
        "dry_run": bool(dry_run),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme_lines = [
        "# RaFT-UAV reproducibility bundle",
        "",
        f"Git SHA: `{manifest['git_sha']}`",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        readme_lines.append(f"### {command.name}")
        if command.description:
            readme_lines.append(command.description)
        readme_lines.append("")
        readme_lines.append("```bash")
        readme_lines.append(" ".join(command.command))
        readme_lines.append("```")
        readme_lines.append("")
        if not dry_run:
            log_path = output_dir / f"{command.name}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                subprocess.run(command.command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return manifest
