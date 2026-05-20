#!/usr/bin/env python3
"""Compare held-out flight distributions with training-flight distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.research.diagnostics import domain_shift_summary, leakage_sentinel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", action="append", type=Path, required=True)
    parser.add_argument("--heldout-csv", type=Path, required=True)
    parser.add_argument("--heldout-flight", default="")
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--columns", nargs="*", default=None)
    args = parser.parse_args(argv)

    train = {str(path): pd.read_csv(path) for path in args.train_csv}
    heldout = pd.read_csv(args.heldout_csv)
    report = domain_shift_summary(train, heldout, columns=args.columns)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_csv, index=False)
    print(f"domain_shift_csv={args.output_csv}")
    if args.metadata_json is not None and args.heldout_flight:
        payload = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        violations = leakage_sentinel(payload, heldout_flight=args.heldout_flight)
        leakage_path = args.output_csv.with_name(f"{args.output_csv.stem}_leakage.json")
        leakage_path.write_text(
            json.dumps([violation.__dict__ for violation in violations], indent=2),
            encoding="utf-8",
        )
        print(f"leakage_json={leakage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
