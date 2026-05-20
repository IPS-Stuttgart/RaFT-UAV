#!/usr/bin/env python3
"""Build a reproducibility bundle for RaFT-UAV paper-style results.

The script is intentionally conservative: it orchestrates existing scripts and
records provenance.  With ``--dry-run`` it only writes the planned commands.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_reproducibility"))
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            sys.executable,
            "scripts/run_sota_readiness_report.py",
            str(args.dataset_root),
            "--output-dir",
            str(args.output_dir / "sota_readiness"),
            "--flights",
            *args.flights,
        ],
        [
            sys.executable,
            "scripts/run_leave_flight_out_sota.py",
            str(args.dataset_root),
            "--output-dir",
            str(args.output_dir / "leave_flight_out"),
            "--flights",
            *args.flights,
        ],
    ]
    if args.skip_existing:
        commands[0].append("--skip-existing")
        commands[1].append("--skip-existing")

    manifest = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "flights": args.flights,
        "commands": [" ".join(command) for command in commands],
        "git_sha": _git_sha(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")
    for command in commands:
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=REPO_ROOT, env=_env())
    return 0


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else src + os.pathsep + env["PYTHONPATH"]
    return env


def _readme(manifest: dict[str, object]) -> str:
    commands = "\n".join(f"```bash\n{command}\n```" for command in manifest["commands"])
    return f"""# RaFT-UAV paper reproducibility bundle

This directory was produced by `scripts/reproduce_paper_results.py`.

Git SHA: `{manifest.get('git_sha')}`

Dataset root: `{manifest['dataset_root']}`

Flights: `{', '.join(manifest['flights'])}`

## Commands

{commands}
"""


if __name__ == "__main__":
    raise SystemExit(main())
