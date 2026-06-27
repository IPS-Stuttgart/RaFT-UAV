"""Train-fit/apply pipeline for MMUAD/UG2+ Track 5 sequence gates.

This module wraps the sequence-gate fitter and applier into one reproducible
command:

1. fit sequence blend weights on a train base/alternate/truth split;
2. predict blend weights for a separate apply split without apply truth;
3. write an official Track 5 CSV/ZIP;
4. optionally score the apply ZIP when public-validation truth is available.

It is intended to make train-selected public-validation and hidden-test package
generation less fragile than a notebook or shell transcript.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.track5_scorecard import build_track5_scorecard, write_track5_scorecard
from raft_uav.mmuad.track5_sequence_gate import blend_track5_sequence_gate
from raft_uav.mmuad.track5_sequence_gate import write_track5_sequence_gate_outputs
from raft_uav.mmuad.track5_sequence_gate_fit import _grid_from_args
from raft_uav.mmuad.track5_sequence_gate_fit import FEATURE_PRESETS
from raft_uav.mmuad.track5_sequence_gate_fit import _jsonable
from raft_uav.mmuad.track5_sequence_gate_fit import _load_track5_gate_rows
from raft_uav.mmuad.track5_sequence_gate_fit import fit_track5_sequence_gate
from raft_uav.mmuad.track5_sequence_gate_fit import write_track5_sequence_gate_fit_outputs
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

FIT_DIR = "sequence_gate_fit"
GATE_DIR = "sequence_gate_apply"
SCORECARD_DIR = "scorecard"
MANIFEST_JSON = "mmuad_track5_sequence_gate_train_apply_manifest.json"
SCORECARD_JSON = "track5_scorecard_sequence_gate_train_apply.json"
SCORECARD_CSV = "track5_scorecard_sequence_gate_train_apply.csv"
POSE_BY_SEQUENCE_CSV = "mmuad_pose_by_sequence_sequence_gate_train_apply.csv"


@dataclass(frozen=True)
class SequenceGateTrainApplyPaths:
    """Artifact paths from the train-fit/apply pipeline."""

    manifest_json: Path
    fit_paths: dict[str, Path]
    gate_paths: dict[str, Path]
    scorecard_paths: dict[str, Path]


def run_track5_sequence_gate_train_apply(
    *,
    train_base_submission_path: Path,
    train_alternate_submission_path: Path,
    train_truth_path: Path,
    apply_base_submission_path: Path,
    apply_alternate_submission_path: Path,
    output_dir: Path,
    apply_truth_path: Path | None = None,
    template_path: Path | None = None,
    models: tuple[str, ...] = (),
    weight_grid: np.ndarray | None = None,
    class_policy: str = "base",
    feature_preset: str = "all",
    protocol: str = "train_fit_apply_not_hidden_test_scored_unless_truth_provided",
    random_state: int = 13,
    require_leaderboard_ready: bool = False,
) -> SequenceGateTrainApplyPaths:
    """Fit on train, apply to another split, and write package artifacts."""

    output = Path(output_dir)
    fit_dir = output / FIT_DIR
    gate_dir = output / GATE_DIR
    scorecard_dir = output / SCORECARD_DIR
    grid = weight_grid if weight_grid is not None else np.linspace(0.0, 0.5, 51)
    selected_models = models or (
        "ridge",
        "tree_d2_leaf1",
        "tree_d3_leaf1",
        "tree_d4_leaf1",
        "rf_depth2",
        "extra_depth2",
    )

    fit_result = fit_track5_sequence_gate(
        base_submission=_load_track5_gate_rows(train_base_submission_path),
        alternate_submission=_load_track5_gate_rows(train_alternate_submission_path),
        truth=_load_track5_gate_rows(train_truth_path),
        apply_base_submission=_load_track5_gate_rows(apply_base_submission_path),
        apply_alternate_submission=_load_track5_gate_rows(apply_alternate_submission_path),
        weight_grid=grid,
        models=tuple(selected_models),
        feature_preset=feature_preset,
        random_state=int(random_state),
    )
    if fit_result.apply_weights is None:
        raise RuntimeError("sequence-gate fitter did not produce apply weights")
    fit_paths = write_track5_sequence_gate_fit_outputs(
        result=fit_result,
        output_dir=fit_dir,
        base_submission_path=train_base_submission_path,
        alternate_submission_path=train_alternate_submission_path,
        truth_path=train_truth_path,
        apply_base_submission_path=apply_base_submission_path,
        apply_alternate_submission_path=apply_alternate_submission_path,
        weight_grid=grid,
        protocol=protocol,
    )

    gate_result = blend_track5_sequence_gate(
        base_submission=load_track5_submission(apply_base_submission_path),
        alternate_submission=load_track5_submission(apply_alternate_submission_path),
        sequence_weights=fit_result.apply_weights,
        default_weight=0.0,
        class_policy=class_policy,  # type: ignore[arg-type]
    )
    template = None if template_path is None else load_official_track5_template_file(template_path)
    gate_paths = write_track5_sequence_gate_outputs(
        result=gate_result,
        output_dir=gate_dir,
        base_submission_path=apply_base_submission_path,
        alternate_submission_path=apply_alternate_submission_path,
        sequence_weights_path=fit_paths["apply_weights_csv"],
        template=template,
        require_leaderboard_ready=template is not None and bool(require_leaderboard_ready),
        manifest={
            "protocol": protocol,
            "fit_summary_json": str(fit_paths["summary_json"]),
            "fit_best_model": fit_result.best_model,
            "feature_preset": fit_result.feature_preset,
            "feature_columns": list(fit_result.feature_columns),
            "train_base_submission": str(train_base_submission_path),
            "train_alternate_submission": str(train_alternate_submission_path),
            "train_truth": str(train_truth_path),
        },
    )

    scorecard_paths: dict[str, Path] = {}
    scorecard_summary: dict[str, Any] | None = None
    if apply_truth_path is not None or template_path is not None:
        scorecard_dir.mkdir(parents=True, exist_ok=True)
        scorecard = build_track5_scorecard(
            results_path=gate_paths["zip"],
            truth_path=apply_truth_path,
            template_path=template_path,
            require_zip=True,
        )
        written_scorecard_paths = write_track5_scorecard(
            scorecard,
            summary_json=scorecard_dir / SCORECARD_JSON,
            summary_csv=scorecard_dir / SCORECARD_CSV,
            pose_by_sequence_csv=scorecard_dir / POSE_BY_SEQUENCE_CSV,
        )
        scorecard_paths = {
            name: Path(path) for name, path in written_scorecard_paths.items()
        }
        scorecard_summary = scorecard.summary
        if require_leaderboard_ready and not scorecard.summary.get(
            "scorecard_leaderboard_ready", False
        ):
            reasons = ", ".join(scorecard.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"sequence-gate train/apply output is not leaderboard-ready: {reasons}")

    manifest = {
        "schema": "raft-uav-mmuad-track5-sequence-gate-train-apply-v1",
        "protocol": protocol,
        "train_base_submission": str(train_base_submission_path),
        "train_alternate_submission": str(train_alternate_submission_path),
        "train_truth": str(train_truth_path),
        "apply_base_submission": str(apply_base_submission_path),
        "apply_alternate_submission": str(apply_alternate_submission_path),
        "apply_truth": None if apply_truth_path is None else str(apply_truth_path),
        "template": None if template_path is None else str(template_path),
        "models": list(selected_models),
        "weight_min": float(np.min(grid)),
        "weight_max": float(np.max(grid)),
        "weight_count": int(len(grid)),
        "feature_preset": fit_result.feature_preset,
        "feature_columns": list(fit_result.feature_columns),
        "feature_count": int(len(fit_result.feature_columns)),
        "best_model": fit_result.best_model,
        "best_fit_row": fit_result.summary.iloc[0].to_dict(),
        "apply_sequence_count": int(fit_result.apply_weights["sequence_id"].nunique()),
        "class_policy": str(class_policy),
        "fit_paths": {name: str(path) for name, path in fit_paths.items()},
        "gate_paths": {name: str(path) for name, path in gate_paths.items()},
        "scorecard_paths": {name: str(path) for name, path in scorecard_paths.items()},
        "scorecard_summary": scorecard_summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / MANIFEST_JSON
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return SequenceGateTrainApplyPaths(
        manifest_json=manifest_path,
        fit_paths=fit_paths,
        gate_paths=gate_paths,
        scorecard_paths=scorecard_paths,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-sequence-gate-train-apply",
        description="fit sequence-gate weights on train and apply them to another Track 5 split",
    )
    parser.add_argument("--train-base-submission", type=Path, required=True)
    parser.add_argument("--train-alternate-submission", type=Path, required=True)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--apply-base-submission", type=Path, required=True)
    parser.add_argument("--apply-alternate-submission", type=Path, required=True)
    parser.add_argument("--apply-truth", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=0.5)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--feature-preset",
        choices=FEATURE_PRESETS,
        default="all",
        help="feature subset used to train/apply the sequence gate",
    )
    parser.add_argument("--class-policy", choices=("base", "alternate"), default="base")
    parser.add_argument("--protocol", default="train_fit_apply_not_hidden_test_scored_unless_truth_provided")
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    paths = run_track5_sequence_gate_train_apply(
        train_base_submission_path=args.train_base_submission,
        train_alternate_submission_path=args.train_alternate_submission,
        train_truth_path=args.train_truth,
        apply_base_submission_path=args.apply_base_submission,
        apply_alternate_submission_path=args.apply_alternate_submission,
        apply_truth_path=args.apply_truth,
        template_path=args.template,
        output_dir=args.output_dir,
        models=tuple(args.model),
        weight_grid=_grid_from_args(args.weight_min, args.weight_max, args.weight_step),
        class_policy=args.class_policy,
        feature_preset=args.feature_preset,
        protocol=args.protocol,
        random_state=args.random_state,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    print("mmuad_track5_sequence_gate_train_apply=ok")
    print(f"manifest_json={paths.manifest_json}")
    print(f"fit_summary_json={paths.fit_paths['summary_json']}")
    print(f"apply_weights_csv={paths.fit_paths['apply_weights_csv']}")
    print(f"ug2_submission_zip={paths.gate_paths['zip']}")
    if "scorecard_json" in paths.scorecard_paths:
        print(f"scorecard_json={paths.scorecard_paths['scorecard_json']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
