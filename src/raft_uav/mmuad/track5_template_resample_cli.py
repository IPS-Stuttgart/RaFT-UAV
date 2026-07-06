"""Text-preserving console wrapper for Track 5 template resampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_template_resample import (
    CLASSIFICATION_POLICIES,
    RESAMPLE_METHODS,
    write_track5_template_resample_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-template-resample",
        description="interpolate MMUAD estimates onto a Track 5 timestamp template",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--resample-method", choices=RESAMPLE_METHODS, default="linear")
    parser.add_argument("--max-interpolation-gap-s", type=float)
    parser.add_argument(
        "--classification-policy",
        choices=CLASSIFICATION_POLICIES,
        default="sequence-mode",
        help=(
            "how to preserve estimate classification values while resampling; "
            "class-map values still override at official CSV/ZIP export time"
        ),
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    estimates = pd.read_csv(args.estimates_csv, dtype=str, keep_default_na=False)
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_template_resample_outputs(
        estimates=estimates,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        resample_method=args.resample_method,
        max_interpolation_gap_s=args.max_interpolation_gap_s,
        classification_policy=args.classification_policy,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_template_resample=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"template-resampled upload is not leaderboard-ready: {reasons}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
