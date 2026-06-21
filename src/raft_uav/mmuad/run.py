"""Convenience entrypoint for MMUAD sequence-root runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.cli import main as track_main

_HELP_FLAGS = {"-h", "--help"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-run",
        description="run the MMUAD tracker on a sequence root",
        add_help=False,
    )
    parser.add_argument("sequence_root", type=Path, nargs="?")
    known, remainder = parser.parse_known_args(argv)
    if known.sequence_root is None:
        if any(arg in _HELP_FLAGS for arg in remainder):
            return track_main(remainder)
        parser.error("the following arguments are required: sequence_root")
    forwarded = ["--sequence-root", str(known.sequence_root), *remainder]
    return track_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
