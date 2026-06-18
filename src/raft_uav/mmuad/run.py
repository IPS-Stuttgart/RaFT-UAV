"""Convenience entrypoint for MMUAD sequence-root runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.cli import main as track_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-run",
        description="run the MMUAD tracker on a sequence root",
        add_help=False,
    )
    parser.add_argument("sequence_root", type=Path)
    known, remainder = parser.parse_known_args(argv)
    forwarded = ["--sequence-root", str(known.sequence_root), *remainder]
    return track_main(forwarded)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
