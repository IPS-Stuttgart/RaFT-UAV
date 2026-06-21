"""Convenience entrypoint for MMUAD sequence-root runs."""

from __future__ import annotations

import argparse
import sys

from raft_uav.mmuad.cli import main as track_main

_HELP_FLAGS = {"-h", "--help"}
_VALUE_FLAGS = {
    "--class-map-csv",
    "--evaluation-json",
    "--evaluate-truth-csv",
    "--evaluate-truth-file",
    "--output-dir",
    "--sequence-classifier",
    "--sequence-glob",
    "--sequence-root",
    "--split-file",
    "--split-name",
    "--ug2-official-codabench-zip",
    "--ug2-official-results-csv",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if _has_explicit_sequence_root(args):
        return track_main(args)

    sequence_root_index = _sequence_root_index(args)
    if sequence_root_index is None:
        if any(arg in _HELP_FLAGS for arg in args):
            return track_main(args)
        parser = argparse.ArgumentParser(
            prog="raft-uav-mmuad-run",
            description="run the MMUAD tracker on a sequence root",
            add_help=False,
        )
        parser.error("the following arguments are required: sequence_root")

    sequence_root = args[sequence_root_index]
    remainder = [arg for index, arg in enumerate(args) if index != sequence_root_index]
    forwarded = ["--sequence-root", sequence_root, *remainder]
    return track_main(forwarded)


def _has_explicit_sequence_root(args: list[str]) -> bool:
    return "--sequence-root" in args or any(arg.startswith("--sequence-root=") for arg in args)


def _sequence_root_index(args: list[str]) -> int | None:
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            return index + 1 if index + 1 < len(args) else None
        if arg in _VALUE_FLAGS:
            skip_next = True
            continue
        if _is_value_flag_assignment(arg):
            continue
        if arg.startswith("-"):
            continue
        return index
    return None


def _is_value_flag_assignment(arg: str) -> bool:
    if "=" not in arg:
        return False
    flag, _value = arg.split("=", 1)
    return flag in _VALUE_FLAGS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
