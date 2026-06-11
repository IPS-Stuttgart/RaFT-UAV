#!/usr/bin/env python
"""Inspect a local MMUAD/UG2+ archive/export layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.layout import inspect_mmuad_layout, write_layout_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/mmuad_layout_report.json"))
    args = parser.parse_args(argv)
    summary = inspect_mmuad_layout(args.root)
    path = write_layout_report(summary, args.output_json)
    print("mmuad_layout_inspection=ok")
    print(f"layout_report_json={path}")
    print(f"file_count={summary['file_count']}")
    print(f"category_counts={summary['category_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
