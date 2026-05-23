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
    parser.add_argument(
        "--paper-flights",
        nargs="*",
        default=["Opt1"],
        help=(
            "flight names for strict paper-parity count/error diagnostics; "
            "kept separate from --flights because Table-II reference counts "
            "are a reproduction target, not a cross-flight SOTA target"
        ),
    )
    parser.add_argument("--skip-paper-parity", action="store_true")
    parser.add_argument(
        "--paper-variant",
        choices=["auto", "original", "rerun"],
        default="auto",
        help="RF/radar/truth file variant passed to paper-strict diagnostics",
    )
    parser.add_argument(
        "--enumerate-paper-file-variants",
        action="store_true",
        help="ask paper-fingerprint to rank original/rerun variants by count deltas",
    )
    parser.add_argument(
        "--paper-count-mismatch-action",
        choices=["ignore", "warn", "fail"],
        default="warn",
        help="how paper-fingerprint/paper-strict should handle Table-II count mismatches",
    )
    parser.add_argument(
        "--paper-enu-origin",
        choices=["truth-first", "lla", "lw1"],
        default="lw1",
        help="paper-parity ENU origin; lw1 requires an explicit LW1 LLA via CLI/env/config",
    )
    parser.add_argument("--paper-enu-origin-lla", default=None)
    parser.add_argument("--paper-lw1-origin-lla", default=None)
    parser.add_argument("--paper-origin-config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    if not args.skip_paper_parity:
        commands.extend(_paper_parity_commands(args))

    sota_readiness_command = [
        sys.executable,
        "scripts/run_sota_readiness_report.py",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir / "sota_readiness"),
        "--flights",
        *args.flights,
    ]
    leave_flight_out_command = [
        sys.executable,
        "scripts/run_leave_flight_out_sota.py",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir / "leave_flight_out"),
        "--flights",
        *args.flights,
    ]
    if args.skip_existing:
        sota_readiness_command.append("--skip-existing")
        leave_flight_out_command.append("--skip-existing")
    commands.extend([sota_readiness_command, leave_flight_out_command])

    manifest = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "flights": args.flights,
        "paper_flights": args.paper_flights,
        "paper_parity_enabled": not args.skip_paper_parity,
        "paper_variant": args.paper_variant,
        "paper_enu_origin": args.paper_enu_origin,
        "paper_count_mismatch_action": args.paper_count_mismatch_action,
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


def _paper_parity_commands(args: argparse.Namespace) -> list[list[str]]:
    """Return strict paper-parity diagnostics that must run before SOTA sweeps."""

    fingerprint = [
        sys.executable,
        "-m",
        "raft_uav.diagnostics.paper_fingerprint",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir / "paper_fingerprint"),
        "--variant",
        args.paper_variant,
        "--count-mismatch-action",
        args.paper_count_mismatch_action,
        "--enu-origin",
        args.paper_enu_origin,
    ]
    if args.enumerate_paper_file_variants:
        fingerprint.append("--enumerate-file-variants")
    _append_repeated_option(fingerprint, "--flight", args.paper_flights)
    _append_optional_option(fingerprint, "--enu-origin-lla", args.paper_enu_origin_lla)
    _append_optional_option(fingerprint, "--lw1-origin-lla", args.paper_lw1_origin_lla)
    _append_optional_option(fingerprint, "--origin-config", args.paper_origin_config)

    strict = [
        sys.executable,
        "-m",
        "raft_uav.diagnostics.paper_strict",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir / "paper_strict"),
        "--variant",
        args.paper_variant,
        "--count-mismatch-action",
        args.paper_count_mismatch_action,
        "--enu-origin",
        args.paper_enu_origin,
    ]
    _append_repeated_option(strict, "--flight", args.paper_flights)
    _append_optional_option(strict, "--enu-origin-lla", args.paper_enu_origin_lla)
    _append_optional_option(strict, "--lw1-origin-lla", args.paper_lw1_origin_lla)
    _append_optional_option(strict, "--origin-config", args.paper_origin_config)

    return [fingerprint, strict]


def _append_repeated_option(command: list[str], option: str, values: list[str]) -> None:
    for value in values:
        command.extend([option, str(value)])


def _append_optional_option(
    command: list[str],
    option: str,
    value: str | Path | None,
) -> None:
    if value is None:
        return
    text = str(value)
    if text:
        command.extend([option, text])


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

Paper-parity flights: `{', '.join(manifest['paper_flights'])}`

Paper-parity enabled: `{manifest['paper_parity_enabled']}`

Paper ENU origin: `{manifest['paper_enu_origin']}`

Paper file variant: `{manifest['paper_variant']}`

Paper count mismatch action: `{manifest['paper_count_mismatch_action']}`

## Commands

{commands}
"""


if __name__ == "__main__":
    raise SystemExit(main())
