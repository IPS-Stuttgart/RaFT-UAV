#!/usr/bin/env python
"""Evaluate train-learned source-coordinate corrections on MMUAD validation.

This is an experiment runner, not a default inference path.  It fits global
source-specific transforms from official train labels, applies those transforms
to validation candidates, then reruns the existing tracker and Track 5
scorecard.  No validation truth is used to fit the transforms.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.evaluator import load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns  # noqa: E402


DEFAULT_WORK_ROOT = Path("/mnt/lexar4tb/mmuad_realdata")
DEFAULT_TRAIN_FEATURES = (
    DEFAULT_WORK_ROOT
    / "outputs/lower_variance_ranker_train_val_20260622_131852/mmuad_train_ranker_features.csv"
)
DEFAULT_VAL_SCORED_CANDIDATES = (
    DEFAULT_WORK_ROOT
    / "outputs/lower_variance_ranker_train_val_20260622_131852/logistic_good5/"
    "mmuad_ranker_scored_candidates.csv"
)
DEFAULT_TRAIN_TRUTH = (
    DEFAULT_WORK_ROOT / "outputs/train_to_val_official_train_20260622_heartbeat/"
    "train_ground_truth_feature_rows.csv"
)
DEFAULT_VAL_TRUTH_NORMALIZED = DEFAULT_WORK_ROOT / (
    "challenge_meta/validation_ref_new_for_your_ref_normalized_truth.csv"
)
DEFAULT_VAL_REFERENCE = DEFAULT_WORK_ROOT / "challenge_meta/validation_ref_new_for_your_ref.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_ROOT / "outputs/mmuad_train_learned_calibration_transfer"

FIT_COLUMNS = (
    "sequence_id",
    "time_s",
    "source",
    "track_id",
    "x_m",
    "y_m",
    "z_m",
    "truth_distance_3d_m",
    "truth_matched",
)
MODES = ("identity", "source-translation", "source-rigid", "source-affine")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--train-truth", type=Path, default=DEFAULT_TRAIN_TRUTH)
    parser.add_argument("--val-scored-candidates", type=Path, default=DEFAULT_VAL_SCORED_CANDIDATES)
    parser.add_argument("--val-truth-normalized", type=Path, default=DEFAULT_VAL_TRUTH_NORMALIZED)
    parser.add_argument("--val-reference", type=Path, default=DEFAULT_VAL_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--max-pair-distance-m", type=float, default=120.0)
    parser.add_argument("--min-pairs-per-source", type=int, default=20)
    parser.add_argument("--selection-confidence-weight", type=float, default=64.0)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    started = time.time()
    truth = load_evaluation_truth_file(args.train_truth).rows
    pairs = _load_fit_pairs(
        args.train_features,
        truth,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        max_pair_distance_m=args.max_pair_distance_m,
    )
    pair_path = args.output_dir / "mmuad_train_learned_calibration_pairs.csv"
    pairs.to_csv(pair_path, index=False)

    scored_candidates = CandidateFrame(
        normalize_candidate_columns(pd.read_csv(args.val_scored_candidates))
    ).rows

    records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    for mode in args.modes:
        run_dir = args.output_dir / _safe_name(mode)
        run_dir.mkdir(exist_ok=True)
        transforms, mode_fit_records = _fit_mode_transforms(
            pairs,
            mode=mode,
            min_pairs_per_source=args.min_pairs_per_source,
        )
        fit_records.extend(mode_fit_records)
        transform_json = run_dir / "source_transforms.json"
        transform_json.write_text(
            json.dumps(
                {
                    "schema": "raft-uav-mmuad-train-learned-source-calibration-v1",
                    "mode": mode,
                    "train_features": str(args.train_features),
                    "train_truth": str(args.train_truth),
                    "fit_pair_csv": str(pair_path),
                    "max_truth_time_delta_s": float(args.max_truth_time_delta_s),
                    "max_pair_distance_m": float(args.max_pair_distance_m),
                    "min_pairs_per_source": int(args.min_pairs_per_source),
                    "transforms": {
                        source: transform.to_jsonable()
                        for source, transform in transforms.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        calibrated = _apply_transforms(scored_candidates, transforms)
        calibrated_path = run_dir / "mmuad_ranker_scored_candidates_calibrated.csv"
        calibrated.to_csv(calibrated_path, index=False)

        tracker_status = _run_tracker(
            calibrated_path,
            args.val_truth_normalized,
            args.val_reference,
            run_dir,
            selection_confidence_weight=args.selection_confidence_weight,
            logs_dir=logs_dir,
        )
        scorecard_status = _run_scorecard(run_dir, args.val_reference, logs_dir=logs_dir)
        scorecard = _read_json(run_dir / "track5_scorecard.json")
        pooled = (scorecard.get("public_track5") or {}).get("pooled") or {}
        nearest = (scorecard.get("nearest_time") or {}).get("pooled") or {}
        records.append(
            {
                "mode": mode,
                "tracker_status": tracker_status,
                "scorecard_status": scorecard_status,
                "pose_mse_loss_m2": pooled.get("pose_mse_loss_m2"),
                "rmse_3d_m": nearest.get("rmse_3d_m"),
                "p95_3d_m": pooled.get("p95_3d_m") or nearest.get("p95_3d_m"),
                "classification_accuracy": pooled.get("classification_accuracy"),
                "fit_pair_count": int(len(pairs)),
                "transformed_source_count": int(len(transforms)),
                "calibrated_candidates_csv": str(calibrated_path),
                "results_csv": str(run_dir / "mmaud_results.csv"),
                "results_zip": str(run_dir / "ug2_submission.zip"),
                "scorecard_json": str(run_dir / "track5_scorecard.json"),
                "source_transforms_json": str(transform_json),
            }
        )
        _write_outputs(args, records, fit_records, pair_path, started)

    _write_outputs(args, records, fit_records, pair_path, started)
    print(f"summary_csv={args.output_dir / 'mmuad_train_learned_calibration_transfer_summary.csv'}")
    print(f"summary_json={args.output_dir / 'mmuad_train_learned_calibration_transfer_summary.json'}")
    return 0


class SourceTransform:
    def __init__(self, linear: np.ndarray, translation: np.ndarray) -> None:
        linear = np.asarray(linear, dtype=float)
        translation = np.asarray(translation, dtype=float).reshape(3)
        if linear.shape != (3, 3):
            raise ValueError(f"linear transform must be 3x3, got {linear.shape}")
        self.linear = linear
        self.translation = translation

    @classmethod
    def identity(cls) -> "SourceTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply(self, xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(xyz, dtype=float)
        return points @ self.linear.T + self.translation

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "linear": self.linear.tolist(),
            "translation_m": self.translation.tolist(),
            "translation_norm_m": float(np.linalg.norm(self.translation)),
        }


def _load_fit_pairs(
    train_features: Path,
    truth: pd.DataFrame,
    *,
    max_truth_time_delta_s: float,
    max_pair_distance_m: float,
) -> pd.DataFrame:
    rows = pd.read_csv(train_features, usecols=lambda col: col in set(FIT_COLUMNS))
    for col in ("time_s", "x_m", "y_m", "z_m", "truth_distance_3d_m"):
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    truth_matched = rows.get("truth_matched", True)
    if not isinstance(truth_matched, pd.Series):
        truth_mask = np.ones(len(rows), dtype=bool)
    else:
        truth_mask = truth_matched.astype(str).str.lower().isin({"true", "1", "yes"})
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m", "truth_distance_3d_m"]]).all(axis=1)
    rows = rows.loc[
        truth_mask
        & finite
        & (rows["truth_distance_3d_m"] <= float(max_pair_distance_m))
    ].copy()
    rows = (
        rows.sort_values("truth_distance_3d_m")
        .drop_duplicates(["sequence_id", "source", "time_s"], keep="first")
        .sort_values(["sequence_id", "source", "time_s"])
        .reset_index(drop=True)
    )

    truth_rows = truth.copy()
    for col in ("time_s", "x_m", "y_m", "z_m"):
        truth_rows[col] = pd.to_numeric(truth_rows[col], errors="coerce")
    truth_rows = truth_rows.loc[
        np.isfinite(truth_rows[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)
    ].copy()

    parts: list[pd.DataFrame] = []
    for sequence, seq_rows in rows.groupby("sequence_id", sort=False):
        seq_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence)].copy()
        if seq_truth.empty:
            continue
        seq_truth = seq_truth.sort_values("time_s")
        seq_rows = seq_rows.sort_values("time_s")
        merged = pd.merge_asof(
            seq_rows,
            seq_truth[["time_s", "x_m", "y_m", "z_m"]].rename(
                columns={"x_m": "truth_x_m", "y_m": "truth_y_m", "z_m": "truth_z_m"}
            ),
            on="time_s",
            direction="nearest",
            tolerance=float(max_truth_time_delta_s),
        )
        parts.append(merged)
    if not parts:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "time_s",
                "source",
                "x_m",
                "y_m",
                "z_m",
                "truth_x_m",
                "truth_y_m",
                "truth_z_m",
                "truth_distance_3d_m",
            ]
        )
    pairs = pd.concat(parts, ignore_index=True)
    finite_truth = np.isfinite(pairs[["truth_x_m", "truth_y_m", "truth_z_m"]]).all(axis=1)
    pairs = pairs.loc[finite_truth].copy()
    before = np.linalg.norm(
        pairs[["x_m", "y_m", "z_m"]].to_numpy(float)
        - pairs[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float),
        axis=1,
    )
    pairs["pair_error_before_m"] = before
    return pairs.reset_index(drop=True)


def _fit_mode_transforms(
    pairs: pd.DataFrame,
    *,
    mode: str,
    min_pairs_per_source: int,
) -> tuple[dict[str, SourceTransform], list[dict[str, Any]]]:
    transforms: dict[str, SourceTransform] = {}
    fit_records: list[dict[str, Any]] = []
    for source, group in pairs.groupby("source", sort=True):
        source_text = str(source)
        x = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        y = group[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
        if len(group) < int(min_pairs_per_source) or mode == "identity":
            transform = SourceTransform.identity()
        elif mode == "source-translation":
            transform = _fit_translation(x, y)
        elif mode == "source-rigid":
            transform = _fit_rigid(x, y)
        elif mode == "source-affine":
            transform = _fit_affine(x, y)
        else:
            raise ValueError(f"unsupported mode: {mode}")
        transforms[source_text] = transform
        after = np.linalg.norm(transform.apply(x) - y, axis=1)
        before = np.linalg.norm(x - y, axis=1)
        fit_records.append(
            {
                "mode": mode,
                "source": source_text,
                "pair_count": int(len(group)),
                "before_mean_m": _mean(before),
                "before_p95_m": _percentile(before, 95.0),
                "after_mean_m": _mean(after),
                "after_p95_m": _percentile(after, 95.0),
                "translation_x_m": float(transform.translation[0]),
                "translation_y_m": float(transform.translation[1]),
                "translation_z_m": float(transform.translation[2]),
                "translation_norm_m": float(np.linalg.norm(transform.translation)),
                "linear_det": float(np.linalg.det(transform.linear)),
            }
        )
    return transforms, fit_records


def _fit_translation(x: np.ndarray, y: np.ndarray) -> SourceTransform:
    residual = y - x
    translation = np.nanmedian(residual, axis=0)
    return SourceTransform(np.eye(3), translation)


def _fit_rigid(x: np.ndarray, y: np.ndarray) -> SourceTransform:
    x_centroid = np.nanmean(x, axis=0)
    y_centroid = np.nanmean(y, axis=0)
    x_centered = x - x_centroid
    y_centered = y - y_centroid
    covariance = x_centered.T @ y_centered
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = y_centroid - rotation @ x_centroid
    return SourceTransform(rotation, translation)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> SourceTransform:
    design = np.column_stack([x, np.ones(len(x), dtype=float)])
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    linear = params[:3, :].T
    translation = params[3, :]
    return SourceTransform(linear, translation)


def _apply_transforms(rows: pd.DataFrame, transforms: dict[str, SourceTransform]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for source, group in rows.groupby("source", sort=False):
        group = group.copy()
        transform = transforms.get(str(source))
        if transform is not None:
            xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
            out = transform.apply(xyz)
            group["x_m"] = out[:, 0]
            group["y_m"] = out[:, 1]
            group["z_m"] = out[:, 2]
            group["train_learned_calibration_applied"] = True
            group["train_learned_calibration_source"] = str(source)
        else:
            group["train_learned_calibration_applied"] = False
            group["train_learned_calibration_source"] = ""
        parts.append(group)
    out = pd.concat(parts, ignore_index=True) if parts else rows.copy()
    return normalize_candidate_columns(out)


def _run_tracker(
    candidates_csv: Path,
    normalized_truth_file: Path,
    official_reference_file: Path,
    run_dir: Path,
    *,
    selection_confidence_weight: float,
    logs_dir: Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.cli",
        "--candidate-csv",
        str(candidates_csv),
        "--truth-file",
        str(normalized_truth_file),
        "--output-dir",
        str(run_dir),
        "--trajectory-completion-mode",
        "fixed-lag",
        "--trajectory-speed-gate-mps",
        "60",
        "--trajectory-outlier-replacement",
        "local-linear",
        "--ug2-official-complete-to-sequence-timestamps",
        "--official-validation-template-file",
        str(official_reference_file),
        "--ug2-official-results-csv",
        str(run_dir / "mmaud_results.csv"),
        "--ug2-official-codabench-zip",
        str(run_dir / "ug2_submission.zip"),
        "--ug2-official-validate-on-write",
        "--official-validation-json",
        str(run_dir / "official_validation.json"),
        "--official-validation-rows-csv",
        str(run_dir / "official_validation_rows.csv"),
        "--official-upload-manifest-json",
        str(run_dir / "official_upload_manifest.json"),
        "--selection-confidence-weight",
        str(selection_confidence_weight),
    ]
    return _run_command(cmd, logs_dir / f"{run_dir.name}_tracker")


def _run_scorecard(run_dir: Path, reference_file: Path, *, logs_dir: Path) -> int:
    cmd = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.track5_scorecard_cli",
        "--results",
        str(run_dir / "ug2_submission.zip"),
        "--truth",
        str(reference_file),
        "--template",
        str(reference_file),
        "--official-upload-manifest",
        str(run_dir / "official_upload_manifest.json"),
        "--output-json",
        str(run_dir / "track5_scorecard.json"),
        "--summary-csv",
        str(run_dir / "track5_scorecard.csv"),
        "--validation-rows-csv",
        str(run_dir / "track5_scorecard_validation_rows.csv"),
        "--public-evaluation-rows-csv",
        str(run_dir / "track5_scorecard_public_rows.csv"),
        "--nearest-time-rows-csv",
        str(run_dir / "track5_scorecard_nearest_rows.csv"),
    ]
    return _run_command(cmd, logs_dir / f"{run_dir.name}_scorecard")


def _run_command(cmd: list[str], log_stem: Path) -> int:
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
    log_stem.with_suffix(".stdout.log").write_text(result.stdout, encoding="utf-8")
    log_stem.with_suffix(".stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return int(result.returncode)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_outputs(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    fit_records: list[dict[str, Any]],
    pair_path: Path,
    started: float,
) -> None:
    summary_csv = args.output_dir / "mmuad_train_learned_calibration_transfer_summary.csv"
    summary_json = args.output_dir / "mmuad_train_learned_calibration_transfer_summary.json"
    fits_csv = args.output_dir / "mmuad_train_learned_source_calibration_fits.csv"
    pd.DataFrame.from_records(records).to_csv(summary_csv, index=False)
    pd.DataFrame.from_records(fit_records).to_csv(fits_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-train-learned-calibration-transfer-v1",
                "protocol": (
                    "Transforms are fit on official train labels only and applied to "
                    "public validation candidates; validation truth is used only for scoring."
                ),
                "train_features": str(args.train_features),
                "train_truth": str(args.train_truth),
                "val_scored_candidates": str(args.val_scored_candidates),
                "val_truth_normalized": str(args.val_truth_normalized),
                "val_reference": str(args.val_reference),
                "fit_pairs_csv": str(pair_path),
                "fit_summary_csv": str(fits_csv),
                "duration_s": round(time.time() - started, 3),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _percentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else float("nan")


def _safe_name(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


if __name__ == "__main__":
    raise SystemExit(main())
