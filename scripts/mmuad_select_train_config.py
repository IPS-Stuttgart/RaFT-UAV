#!/usr/bin/env python
"""Select a frozen MMUAD train-to-validation config using train-only LOSO.

This script is infrastructure for defensible MMUAD paper rows: it tunes method
settings using only official train labels, writes ``mmuad_train_selected_config``
artifacts, and can optionally run the public-validation harness once with that
frozen config.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.classification import (  # noqa: E402
    build_sequence_classifier_loso_predictions,
    load_sequence_class_labels,
    sequence_features_from_sequence_root,
    write_sequence_classifier_loso_predictions,
)
from raft_uav.mmuad.cluster_ranker import (  # noqa: E402
    _load_candidates_from_args,
    build_cluster_feature_table,
    predict_cluster_scores,
    train_cluster_ranker,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.schema import CandidateFrame, TruthFrame, normalize_candidate_columns  # noqa: E402
from raft_uav.mmuad.source_calibration import (  # noqa: E402
    apply_source_calibration_payload,
    build_source_calibration_pairs,
    fit_source_calibration,
)
from raft_uav.mmuad.tracker import TrackerConfig, compute_metrics, run_mmuad_tracker  # noqa: E402
from raft_uav.mmuad.train_selected_config import write_train_selected_config  # noqa: E402
from raft_uav.mmuad.trajectory_completion import (  # noqa: E402
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
)


DEFAULT_OUTPUT_DIR = Path("outputs/mmuad_train_selected_config")
DEFAULT_ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_RANKER_SPECS = (
    "sklearn-logistic:good_cluster_5m",
    "random-forest-classifier:good_cluster_5m",
    "random-forest-classifier:good_cluster_10m",
)
DEFAULT_VITERBI_MOTION_WEIGHTS = (1.0, 4.0)
DEFAULT_VITERBI_RANKER_WEIGHTS = (1.0, 2.0)
DEFAULT_VITERBI_SOURCE_SWITCH_PENALTIES = (0.0, 0.25)
DEFAULT_VITERBI_MAX_SPEEDS_MPS = (40.0, 60.0)
DEFAULT_SMOOTHING_MODES = ("none", "fixed-lag")
DEFAULT_SMOOTHING_SPEED_GATES_MPS = (0.0, 20.0)
DEFAULT_SMOOTHING_BLENDS = (0.5, 1.0)
DEFAULT_CLASSIFIER_METHODS = ("random-forest", "hist-gradient-boosting", "nearest-centroid")
DEFAULT_FUSION_WEIGHTS = (0.0,)


@dataclass(frozen=True)
class RankerSpec:
    model_type: str
    target_column: str

    @property
    def run(self) -> str:
        return f"{self.model_type}_{self.target_column}".replace("-", "_")


@dataclass(frozen=True)
class ViterbiSpec:
    motion_weight: float
    ranker_weight: float
    source_switch_penalty: float
    max_speed_mps: float

    @property
    def run(self) -> str:
        return (
            f"motion{self.motion_weight:g}_ranker{self.ranker_weight:g}_"
            f"switch{self.source_switch_penalty:g}_speed{self.max_speed_mps:g}"
        ).replace(".", "p")


@dataclass(frozen=True)
class SmoothingSpec:
    mode: str
    speed_gate_mps: float
    blend: float

    @property
    def run(self) -> str:
        return (
            f"{self.mode}_speed{self.speed_gate_mps:g}_blend{self.blend:g}"
        ).replace("-", "_").replace(".", "p")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--train-candidates", type=Path)
    parser.add_argument("--val-root", type=Path)
    parser.add_argument("--val-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-output-dir", type=Path)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--source-alpha-grid", default=_join(DEFAULT_ALPHA_GRID))
    parser.add_argument("--ranker-spec", action="append", default=[])
    parser.add_argument("--viterbi-motion-weights", default=_join(DEFAULT_VITERBI_MOTION_WEIGHTS))
    parser.add_argument("--viterbi-ranker-weights", default=_join(DEFAULT_VITERBI_RANKER_WEIGHTS))
    parser.add_argument(
        "--viterbi-source-switch-penalties",
        default=_join(DEFAULT_VITERBI_SOURCE_SWITCH_PENALTIES),
    )
    parser.add_argument("--viterbi-max-speeds-mps", default=_join(DEFAULT_VITERBI_MAX_SPEEDS_MPS))
    parser.add_argument("--smoothing-modes", default=",".join(DEFAULT_SMOOTHING_MODES))
    parser.add_argument("--smoothing-speed-gates-mps", default=_join(DEFAULT_SMOOTHING_SPEED_GATES_MPS))
    parser.add_argument("--smoothing-blends", default=_join(DEFAULT_SMOOTHING_BLENDS))
    parser.add_argument("--classifier-method", action="append", default=[])
    parser.add_argument("--classifier-fusion-weights", default=_join(DEFAULT_FUSION_WEIGHTS))
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--max-pair-distance-m", type=float, default=120.0)
    parser.add_argument("--min-pairs-per-source", type=int, default=20)
    parser.add_argument("--selection-confidence-weight", type=float, default=64.0)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--run-public-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = _artifact_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    config_json = paths["selected_config_json"]
    summary_csv = paths["selected_config_summary_csv"]

    if args.dry_run:
        plan = _dry_run_plan(args, config_json)
        _write_json(paths["selector_summary_json"], {"status": "dry_run", "plan": plan})
        print(f"mmuad_train_config_selection_plan={paths['selector_summary_json']}")
        return 0

    candidates = _load_train_candidates(args)
    truth = load_evaluation_truth_file(args.train_truth).rows
    labels = load_sequence_class_labels(args.train_reference)

    source_rows = select_source_alpha_loso(candidates, truth, args)
    source_rows.to_csv(paths["source_alpha_loso_csv"], index=False)
    best_source = _best_by(source_rows, "train_loso_pose_mse_loss_m2", minimize=True)
    source_mode = str(best_source["source_calibration_mode"])
    source_alpha = float(best_source["source_translation_alpha"])

    ranker_rows, best_ranker_predictions = select_ranker_loso(
        candidates,
        truth,
        source_mode=source_mode,
        source_alpha=source_alpha,
        args=args,
    )
    ranker_rows.to_csv(paths["ranker_loso_grid_csv"], index=False)
    best_ranker_predictions.to_csv(paths["ranker_loso_predictions_csv"], index=False)
    best_ranker = _best_by(ranker_rows, "train_loso_pose_mse_loss_m2", minimize=True)

    viterbi_rows, best_viterbi_output = select_viterbi_loso(
        best_ranker_predictions,
        truth,
        args=args,
    )
    viterbi_rows.to_csv(paths["viterbi_loso_grid_csv"], index=False)
    best_viterbi = _best_by(viterbi_rows, "train_loso_pose_mse_loss_m2", minimize=True)

    smoothing_rows = select_smoothing_loso(
        best_viterbi_output.estimates,
        truth,
        args=args,
    )
    smoothing_rows.to_csv(paths["smoothing_loso_grid_csv"], index=False)
    best_smoothing = _best_by(smoothing_rows, "train_loso_pose_mse_loss_m2", minimize=True)

    classifier_rows, classifier_predictions = select_classifier_loso(args, labels)
    classifier_rows.to_csv(paths["classifier_loso_grid_csv"], index=False)
    write_sequence_classifier_loso_predictions(
        classifier_predictions,
        paths["classifier_loso_predictions_csv"],
    )
    best_classifier = _best_by(classifier_rows, "train_loso_classification_accuracy", minimize=False)

    selected_config = {
        "source_calibration_mode": source_mode,
        "source_translation_alpha": source_alpha,
        "ranker_model_type": best_ranker["ranker_model_type"],
        "ranker_target_column": best_ranker["ranker_target_column"],
        "mmuad_selection_mode": "viterbi",
        "viterbi_motion_weight": best_viterbi["viterbi_motion_weight"],
        "viterbi_ranker_weight": best_viterbi["viterbi_ranker_weight"],
        "viterbi_source_switch_penalty": best_viterbi["viterbi_source_switch_penalty"],
        "viterbi_max_speed_mps": best_viterbi["viterbi_max_speed_mps"],
        "viterbi_gap_penalty": 0.0,
        "smoothing_mode": best_smoothing["smoothing_mode"],
        "smoothing_speed_gate_mps": best_smoothing["smoothing_speed_gate_mps"],
        "smoothing_blend": best_smoothing["smoothing_blend"],
        "classifier_method": best_classifier["classifier_method"],
        "image_nonimage_fusion_weight": best_classifier["image_nonimage_fusion_weight"],
    }
    write_train_selected_config(
        selected_config,
        output_json=config_json,
        summary_csv=summary_csv,
        selection_records=[
            _record_from_row("source_calibration", best_source),
            _record_from_row("ranker", best_ranker),
            _record_from_row("viterbi", best_viterbi),
            _record_from_row("smoothing", best_smoothing),
            _record_from_row("classifier", best_classifier),
        ],
        selection_inputs={
            "protocol": "train_loso_only",
            "train_root": str(args.train_root),
            "train_truth": str(args.train_truth),
            "train_reference": str(args.train_reference),
        },
    )
    selector_summary = {
        "schema": "raft-uav-mmuad-train-loso-config-selector-v1",
        "protocol": "train_loso_selection_then_optional_single_public_validation_eval",
        "selected_config_json": str(config_json),
        "selected_config_summary_csv": str(summary_csv),
        "source_alpha_loso_csv": str(paths["source_alpha_loso_csv"]),
        "ranker_loso_grid_csv": str(paths["ranker_loso_grid_csv"]),
        "viterbi_loso_grid_csv": str(paths["viterbi_loso_grid_csv"]),
        "smoothing_loso_grid_csv": str(paths["smoothing_loso_grid_csv"]),
        "classifier_loso_grid_csv": str(paths["classifier_loso_grid_csv"]),
        "selected_config": selected_config,
    }

    if args.run_public_validation:
        selector_summary["public_validation_command"] = _run_public_validation(
            args,
            config_json,
        )
    _write_json(paths["selector_summary_json"], selector_summary)
    print("mmuad_train_config_selection=ok")
    print(f"selected_config_json={config_json}")
    print(f"selected_config_summary_csv={summary_csv}")
    return 0


def select_source_alpha_loso(
    candidates: CandidateFrame,
    truth: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pairs = build_source_calibration_pairs(
        candidates,
        truth,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        max_pair_distance_m=args.max_pair_distance_m,
    )
    alphas = _float_list(args.source_alpha_grid)
    records: list[dict[str, Any]] = []
    for alpha in alphas:
        squared_errors: list[float] = []
        for _source, source_rows in pairs.groupby("source", sort=True):
            sequences = sorted(source_rows["sequence_id"].astype(str).unique())
            for heldout in sequences:
                train_rows = source_rows.loc[source_rows["sequence_id"].astype(str) != heldout]
                heldout_rows = source_rows.loc[source_rows["sequence_id"].astype(str) == heldout]
                if train_rows.empty or heldout_rows.empty:
                    continue
                translation = np.nanmedian(
                    train_rows[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
                    - train_rows[["x_m", "y_m", "z_m"]].to_numpy(float),
                    axis=0,
                )
                residual = (
                    heldout_rows[["x_m", "y_m", "z_m"]].to_numpy(float)
                    + float(alpha) * translation
                    - heldout_rows[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
                )
                squared_errors.extend(np.sum(residual**2, axis=1).tolist())
        mse = _safe_mean(squared_errors)
        records.append(
            {
                "source_calibration_mode": "identity" if float(alpha) == 0.0 else "source-translation",
                "source_translation_alpha": float(alpha),
                "train_loso_pair_count": int(len(squared_errors)),
                "train_loso_pose_mse_loss_m2": mse,
                "train_loso_pose_rmse_m": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["train_loso_pose_mse_loss_m2", "source_translation_alpha"]
    )


def select_ranker_loso(
    candidates: CandidateFrame,
    truth: pd.DataFrame,
    *,
    source_mode: str,
    source_alpha: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = _ranker_specs(args.ranker_spec)
    records: list[dict[str, Any]] = []
    predictions_by_run: dict[str, pd.DataFrame] = {}
    for spec in specs:
        predictions = _ranker_loso_predictions(
            candidates,
            truth,
            source_mode=source_mode,
            source_alpha=source_alpha,
            spec=spec,
            args=args,
        )
        predictions_by_run[spec.run] = predictions
        summary = _candidate_top1_summary(predictions)
        records.append(
            {
                "ranker_run": spec.run,
                "ranker_model_type": spec.model_type,
                "ranker_target_column": spec.target_column,
                **summary,
            }
        )
    frame = pd.DataFrame.from_records(records).sort_values(
        ["train_loso_pose_mse_loss_m2", "ranker_run"]
    )
    best_run = str(frame.iloc[0]["ranker_run"])
    return frame, predictions_by_run[best_run]


def _ranker_loso_predictions(
    candidates: CandidateFrame,
    truth: pd.DataFrame,
    *,
    source_mode: str,
    source_alpha: float,
    spec: RankerSpec,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = candidates.rows.copy()
    truth_rows = truth.copy()
    sequence_ids = sorted(rows["sequence_id"].astype(str).unique())
    predictions: list[pd.DataFrame] = []
    for heldout in sequence_ids:
        train_candidates = CandidateFrame(
            normalize_candidate_columns(rows.loc[rows["sequence_id"].astype(str) != heldout])
        )
        heldout_candidates = CandidateFrame(
            normalize_candidate_columns(rows.loc[rows["sequence_id"].astype(str) == heldout])
        )
        train_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) != heldout].copy()
        heldout_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == heldout].copy()
        if train_candidates.rows.empty or heldout_candidates.rows.empty or train_truth.empty:
            continue
        if source_mode != "identity":
            payload, _pairs, _summary = fit_source_calibration(
                train_candidates,
                train_truth,
                mode="source-translation",
                max_truth_time_delta_s=args.max_truth_time_delta_s,
                max_pair_distance_m=args.max_pair_distance_m,
                min_pairs_per_source=args.min_pairs_per_source,
                source_translation_alpha_grid=[source_alpha],
            )
            train_candidates = apply_source_calibration_payload(train_candidates, payload)
            heldout_candidates = apply_source_calibration_payload(heldout_candidates, payload)
        train_features = build_cluster_feature_table(
            train_candidates,
            truth=train_truth,
            good_threshold_m=args.good_threshold_m,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        heldout_features = build_cluster_feature_table(
            heldout_candidates,
            truth=heldout_truth,
            good_threshold_m=args.good_threshold_m,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        model = train_cluster_ranker(
            train_features,
            model_type=spec.model_type,
            target_column=spec.target_column,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
            score_distance_scale_m=args.good_threshold_m,
        )
        heldout_features["ranker_score"] = predict_cluster_scores(heldout_features, model)
        heldout_features["confidence"] = heldout_features["ranker_score"]
        heldout_features["loso_heldout_sequence"] = heldout
        heldout_features["ranker_run"] = spec.run
        heldout_features["ranker_model_type"] = spec.model_type
        heldout_features["ranker_target_column"] = spec.target_column
        predictions.append(heldout_features)
    if not predictions:
        raise ValueError(f"no ranker LOSO folds produced predictions for {spec.run}")
    return pd.concat(predictions, ignore_index=True)


def select_viterbi_loso(
    scored_candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    args: argparse.Namespace,
):
    candidates = CandidateFrame(normalize_candidate_columns(scored_candidates))
    truth_frame = TruthFrame(truth)
    records: list[dict[str, Any]] = []
    outputs = {}
    for spec in _viterbi_specs(args):
        output = run_mmuad_tracker(
            candidates,
            truth_frame,
            config=TrackerConfig(
                selection_mode="viterbi",
                selection_confidence_weight=args.selection_confidence_weight,
                viterbi_motion_weight=spec.motion_weight,
                viterbi_ranker_weight=spec.ranker_weight,
                viterbi_source_switch_penalty=spec.source_switch_penalty,
                viterbi_max_speed_mps=spec.max_speed_mps,
            ),
        )
        outputs[spec.run] = output
        metrics = compute_metrics(output.estimates, truth)
        records.append(
            {
                "viterbi_run": spec.run,
                "mmuad_selection_mode": "viterbi",
                "viterbi_motion_weight": spec.motion_weight,
                "viterbi_ranker_weight": spec.ranker_weight,
                "viterbi_source_switch_penalty": spec.source_switch_penalty,
                "viterbi_max_speed_mps": spec.max_speed_mps,
                "train_loso_pose_mse_loss_m2": _mse_from_estimates(output.estimates),
                "train_loso_pose_rmse_m": metrics.get("rmse_3d_m"),
                "train_loso_p95_3d_m": metrics.get("p95_3d_m"),
                "train_loso_max_3d_m": metrics.get("max_3d_m"),
            }
        )
    frame = pd.DataFrame.from_records(records).sort_values(
        ["train_loso_pose_mse_loss_m2", "train_loso_p95_3d_m", "viterbi_run"]
    )
    return frame, outputs[str(frame.iloc[0]["viterbi_run"])]


def select_smoothing_loso(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in _smoothing_specs(args):
        if spec.mode == "none":
            smoothed = estimates
        else:
            smoothed = complete_and_smooth_estimates(
                estimates,
                truth,
                config=TrajectoryCompletionConfig(
                    mode=spec.mode,
                    speed_gate_mps=spec.speed_gate_mps,
                    smoothing_blend=spec.blend,
                    outlier_replacement="local-linear" if spec.speed_gate_mps > 0 else "none",
                ),
            ).estimates
        metrics = compute_metrics(smoothed, truth)
        records.append(
            {
                "smoothing_run": spec.run,
                "smoothing_mode": spec.mode,
                "smoothing_speed_gate_mps": spec.speed_gate_mps,
                "smoothing_blend": spec.blend,
                "train_loso_pose_mse_loss_m2": _mse_from_estimates(smoothed),
                "train_loso_pose_rmse_m": metrics.get("rmse_3d_m"),
                "train_loso_p95_3d_m": metrics.get("p95_3d_m"),
                "train_loso_max_3d_m": metrics.get("max_3d_m"),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["train_loso_pose_mse_loss_m2", "train_loso_p95_3d_m", "smoothing_run"]
    )


def select_classifier_loso(
    args: argparse.Namespace,
    labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = sequence_features_from_sequence_root(
        args.train_root,
        sequence_glob=args.sequence_glob,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
    )
    methods = args.classifier_method or list(DEFAULT_CLASSIFIER_METHODS)
    fusion_weights = _float_list(args.classifier_fusion_weights)
    records: list[dict[str, Any]] = []
    predictions_by_method: dict[str, pd.DataFrame] = {}
    for method in methods:
        predictions = build_sequence_classifier_loso_predictions(
            features=features,
            labels=labels,
            method=method,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        predictions_by_method[method] = predictions
        accuracy = float(predictions["correct"].astype(bool).mean())
        for weight in fusion_weights:
            records.append(
                {
                    "classifier_method": method,
                    "image_nonimage_fusion_weight": float(weight),
                    "train_loso_classification_accuracy": accuracy,
                    "train_loso_sequence_count": int(len(predictions)),
                    "image_features_available": False,
                }
            )
    frame = pd.DataFrame.from_records(records).sort_values(
        ["train_loso_classification_accuracy", "classifier_method"],
        ascending=[False, True],
    )
    best_method = str(frame.iloc[0]["classifier_method"])
    return frame, predictions_by_method[best_method]


def _run_public_validation(args: argparse.Namespace, config_json: Path) -> list[str]:
    if args.val_root is None or args.val_reference is None:
        raise ValueError("--run-public-validation requires --val-root and --val-reference")
    output_dir = args.validation_output_dir or args.output_dir / "public_validation"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_mmuad_train_to_val_experiment.py"),
        "--train-root",
        str(args.train_root),
        "--train-truth",
        str(args.train_truth),
        "--train-reference",
        str(args.train_reference),
        "--val-root",
        str(args.val_root),
        "--val-reference",
        str(args.val_reference),
        "--selected-config-json",
        str(config_json),
        "--output-dir",
        str(output_dir),
        "--sequence-glob",
        args.sequence_glob,
        "--voxel-size-m",
        str(args.voxel_size_m),
        "--min-cluster-points",
        str(args.min_cluster_points),
    ]
    if args.no_apply_calibration:
        cmd.append("--no-apply-calibration")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    logs_dir = args.output_dir / "logs"
    (logs_dir / "public_validation.stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs_dir / "public_validation.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return cmd


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "selected_config_json": output_dir / "mmuad_train_selected_config.json",
        "selected_config_summary_csv": output_dir / "mmuad_train_selected_config_summary.csv",
        "selector_summary_json": output_dir / "mmuad_train_config_selector_summary.json",
        "source_alpha_loso_csv": output_dir / "mmuad_source_alpha_loso_selection.csv",
        "ranker_loso_grid_csv": output_dir / "mmuad_ranker_loso_selection_grid.csv",
        "ranker_loso_predictions_csv": output_dir / "mmuad_ranker_loso_predictions.csv",
        "viterbi_loso_grid_csv": output_dir / "mmuad_viterbi_loso_selection_grid.csv",
        "smoothing_loso_grid_csv": output_dir / "mmuad_smoothing_loso_selection_grid.csv",
        "classifier_loso_grid_csv": output_dir / "mmuad_classifier_loso_selection_grid.csv",
        "classifier_loso_predictions_csv": output_dir / "mmuad_classifier_loso_predictions.csv",
    }


def _load_train_candidates(args: argparse.Namespace) -> CandidateFrame:
    return _load_candidates_from_args(
        csv_path=args.train_candidates,
        sequence_root=None if args.train_candidates is not None else args.train_root,
        sequence_glob=args.sequence_glob,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
    )


def _candidate_top1_summary(rows: pd.DataFrame) -> dict[str, Any]:
    distances: list[float] = []
    work = rows.copy()
    work["truth_distance_3d_m"] = pd.to_numeric(work["truth_distance_3d_m"], errors="coerce")
    work["ranker_score"] = pd.to_numeric(work["ranker_score"], errors="coerce")
    for (_sequence, _time_s), group in work.groupby(["sequence_id", "time_s"], sort=True):
        group = group.loc[group["truth_distance_3d_m"].notna()]
        if group.empty:
            continue
        selected = group.sort_values("ranker_score", ascending=False).iloc[0]
        distances.append(float(selected["truth_distance_3d_m"]))
    values = np.asarray(distances, dtype=float)
    return {
        "train_loso_frame_count": int(len(values)),
        "train_loso_pose_mse_loss_m2": float(np.mean(values**2)) if values.size else np.nan,
        "train_loso_pose_rmse_m": float(np.sqrt(np.mean(values**2))) if values.size else np.nan,
        "train_loso_mean_3d_m": float(np.mean(values)) if values.size else np.nan,
        "train_loso_p95_3d_m": float(np.percentile(values, 95.0)) if values.size else np.nan,
        "train_loso_max_3d_m": float(np.max(values)) if values.size else np.nan,
    }


def _mse_from_estimates(estimates: pd.DataFrame) -> float:
    if estimates.empty or "error_3d_m" not in estimates.columns:
        return float("nan")
    values = pd.to_numeric(estimates["error_3d_m"], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    return float(np.mean(values**2)) if values.size else float("nan")


def _ranker_specs(values: list[str]) -> list[RankerSpec]:
    specs = values or list(DEFAULT_RANKER_SPECS)
    out = []
    for value in specs:
        if ":" not in value:
            raise ValueError("--ranker-spec must be MODEL_TYPE:TARGET_COLUMN")
        model_type, target_column = value.split(":", 1)
        out.append(RankerSpec(model_type.strip(), target_column.strip()))
    return out


def _viterbi_specs(args: argparse.Namespace) -> list[ViterbiSpec]:
    return [
        ViterbiSpec(motion, ranker, switch, speed)
        for motion in _float_list(args.viterbi_motion_weights)
        for ranker in _float_list(args.viterbi_ranker_weights)
        for switch in _float_list(args.viterbi_source_switch_penalties)
        for speed in _float_list(args.viterbi_max_speeds_mps)
    ]


def _smoothing_specs(args: argparse.Namespace) -> list[SmoothingSpec]:
    return [
        SmoothingSpec(mode, speed, blend)
        for mode in _string_list(args.smoothing_modes)
        for speed in _float_list(args.smoothing_speed_gates_mps)
        for blend in _float_list(args.smoothing_blends)
    ]


def _best_by(rows: pd.DataFrame, column: str, *, minimize: bool) -> pd.Series:
    values = pd.to_numeric(rows[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        raise ValueError(f"no finite values in {column}")
    return rows.loc[int(finite.idxmin() if minimize else finite.idxmax())]


def _record_from_row(component: str, row: pd.Series) -> dict[str, Any]:
    return {"component": component, **{str(key): _jsonable(value) for key, value in row.items()}}


def _dry_run_plan(args: argparse.Namespace, config_json: Path) -> dict[str, Any]:
    plan = {
        "train_root": str(args.train_root),
        "train_truth": str(args.train_truth),
        "train_reference": str(args.train_reference),
        "selected_config_json": str(config_json),
        "source_alpha_grid": _float_list(args.source_alpha_grid),
        "ranker_specs": [spec.__dict__ for spec in _ranker_specs(args.ranker_spec)],
        "viterbi_specs": [spec.__dict__ for spec in _viterbi_specs(args)],
        "smoothing_specs": [spec.__dict__ for spec in _smoothing_specs(args)],
        "classifier_methods": args.classifier_method or list(DEFAULT_CLASSIFIER_METHODS),
        "classifier_fusion_weights": _float_list(args.classifier_fusion_weights),
        "run_public_validation": bool(args.run_public_validation),
    }
    if args.run_public_validation:
        plan["public_validation_requires"] = ["--val-root", "--val-reference"]
    return plan


def _float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _string_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _join(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)


def _safe_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
