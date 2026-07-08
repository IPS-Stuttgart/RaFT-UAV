"""Text-preserving console wrapper for Track 5 calibration shrinkage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_calibration_shrinkage import _parse_alpha_grid
from raft_uav.mmuad.track5_estimate_calibration_shrinkage import (
    write_track5_estimate_calibration_shrinkage_outputs,
)
from raft_uav.mmuad.track5_estimate_calibration_shrinkage import (
    write_track5_estimate_calibration_shrinkage_search_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-calibration-shrinkage",
        description="apply or train-select shrinkage for fitted Track 5 estimate calibration",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--write-apply", action="store_true")
    parser.add_argument("--use-best-alpha", action="store_true")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates = read_estimate_csv(args.estimates_csv)
    template = load_official_track5_template_file(args.template)
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    alpha = float(args.alpha)
    if args.truth_csv is not None:
        search_paths = write_track5_estimate_calibration_shrinkage_search_outputs(
            estimates=estimates,
            template=template,
            truth=load_evaluation_truth_file(args.truth_csv).rows,
            calibration=calibration,
            output_dir=output / "search",
            alpha_values=_parse_alpha_grid(args.alpha_grid),
            max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        )
        paths.update(search_paths)
        if args.use_best_alpha:
            alpha = float(
                json.loads(search_paths["best_alpha_json"].read_text(encoding="utf-8"))["alpha"]
            )
    if args.write_apply or args.truth_csv is None:
        class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
        paths.update(
            write_track5_estimate_calibration_shrinkage_outputs(
                estimates=estimates,
                template=template,
                calibration=calibration,
                output_dir=output / "apply",
                alpha=alpha,
                class_map=class_map,
                default_classification=args.default_classification,
                max_nearest_time_delta_s=args.max_nearest_time_delta_s,
            )
        )
    print("mmuad_track5_calibration_shrinkage=ok")
    print(f"alpha={alpha}")
    for name, path in paths.items():
        print(f"{name}={path}")
    if args.require_leaderboard_ready:
        validation_path = paths.get("validation_json")
        if validation_path is None:
            raise SystemExit("leaderboard readiness requires --write-apply or apply-only mode")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("leaderboard_ready", False):
            reasons = ", ".join(validation.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"shrunk calibrated upload is not leaderboard-ready: {reasons}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
