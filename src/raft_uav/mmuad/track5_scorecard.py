"""One-stop local scorecard for UG2+/MMUAD Track 5 submissions.

The helpers in this module deliberately combine the existing local evaluator
and submission preflight validator.  They are intended for reproducibility and
leaderboard-readiness checks before a Codabench upload.  They do not claim to
be the closed Codabench runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_file,
)
from raft_uav.mmuad.schema import load_jsonable
from raft_uav.mmuad.submission import (
    OfficialTrack5Validation,
    load_official_track5_template_file,
    validate_official_track5_submission,
    verify_official_upload_manifest,
)


@dataclass(frozen=True)
class Track5Scorecard:
    """Combined validation/evaluation payload for one Track 5 result file."""

    summary: dict[str, Any]
    validation_rows: pd.DataFrame
    public_evaluation_rows: pd.DataFrame
    nearest_time_rows: pd.DataFrame


def build_track5_scorecard(
    *,
    results_path: Path,
    truth_path: Path | None = None,
    template_path: Path | None = None,
    class_map_path: Path | None = None,
    upload_manifest_path: Path | None = None,
    require_zip: bool = True,
    timestamp_tolerance_s: float = 1.0e-6,
    max_time_delta_s: float = 0.5,
) -> Track5Scorecard:
    """Validate and locally evaluate a UG2+/MMUAD Track 5 result file.

    Parameters
    ----------
    results_path:
        Path to either an official-style ``mmaud_results.csv`` or a ZIP that
        contains a root-level ``mmaud_results.csv``.
    truth_path:
        Optional normalized or official truth file.  If supplied, the scorecard
        runs both ``public-track5`` timestamp-aligned metrics and nearest-time
        diagnostics.
    template_path:
        Optional official result/template CSV or ZIP.  Validation uses only the
        requested ``Sequence``/``Timestamp`` grid.  When omitted and ``truth`` is
        supplied, the normalized truth timestamps become the validation template.
    class_map_path:
        Optional sequence-to-class map for evaluation.
    upload_manifest_path:
        Optional ``mmuad_official_upload_manifest.json`` written by the
        official validation path.  When supplied, scorecard readiness also
        requires the manifest fingerprints to match the current artifact.
    require_zip:
        Whether validation should require a ZIP.  Keep this true for Codabench
        preflight; set false for local CSV development checks.
    """

    results_path = Path(results_path)
    truth_path = Path(truth_path) if truth_path is not None else None
    template_path = Path(template_path) if template_path is not None else None
    class_map_path = Path(class_map_path) if class_map_path is not None else None
    upload_manifest_path = (
        Path(upload_manifest_path) if upload_manifest_path is not None else None
    )

    truth_frame = load_evaluation_truth_file(truth_path) if truth_path is not None else None
    template = _scorecard_template_frame(truth_frame=truth_frame, template_path=template_path)
    validation = validate_official_track5_submission(
        results_path,
        template=template,
        timestamp_tolerance_s=timestamp_tolerance_s,
        require_zip=require_zip,
    )

    public_eval: dict[str, Any] | None = None
    nearest_eval: dict[str, Any] | None = None
    manifest_verification: dict[str, Any] | None = None
    public_rows = pd.DataFrame()
    nearest_rows = pd.DataFrame()
    if truth_frame is not None:
        results = load_mmaud_results_file(results_path)
        public_eval = evaluate_mmaud_results(
            results,
            truth_frame,
            metric_protocol="public-track5",
            timestamp_tolerance_s=timestamp_tolerance_s,
            class_map_path=class_map_path,
        )
        nearest_eval = evaluate_mmaud_results(
            results,
            truth_frame,
            metric_protocol="nearest-time",
            max_time_delta_s=max_time_delta_s,
            class_map_path=class_map_path,
        )
        public_rows = public_eval["rows"]
        nearest_rows = nearest_eval["rows"]
    if upload_manifest_path is not None:
        manifest_verification = verify_official_upload_manifest(upload_manifest_path)

    summary = _scorecard_summary(
        results_path=results_path,
        truth_path=truth_path,
        template_path=template_path,
        class_map_path=class_map_path,
        upload_manifest_path=upload_manifest_path,
        require_zip=require_zip,
        timestamp_tolerance_s=timestamp_tolerance_s,
        max_time_delta_s=max_time_delta_s,
        validation=validation,
        public_eval=public_eval,
        nearest_eval=nearest_eval,
        manifest_verification=manifest_verification,
    )
    return Track5Scorecard(
        summary=summary,
        validation_rows=validation.rows,
        public_evaluation_rows=public_rows,
        nearest_time_rows=nearest_rows,
    )


def write_track5_scorecard(
    scorecard: Track5Scorecard,
    *,
    summary_json: Path,
    summary_csv: Path | None = None,
    validation_rows_csv: Path | None = None,
    public_evaluation_rows_csv: Path | None = None,
    nearest_time_rows_csv: Path | None = None,
) -> dict[str, str]:
    """Write scorecard artifacts and return path labels."""

    summary_json = Path(summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(load_jsonable(scorecard.summary), indent=2),
        encoding="utf-8",
    )
    paths = {"scorecard_json": str(summary_json)}
    if summary_csv is not None:
        summary_csv = Path(summary_csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard_summary_frame(scorecard.summary).to_csv(summary_csv, index=False)
        paths["scorecard_csv"] = str(summary_csv)
    if validation_rows_csv is not None:
        validation_rows_csv = Path(validation_rows_csv)
        validation_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard.validation_rows.to_csv(validation_rows_csv, index=False)
        paths["validation_rows_csv"] = str(validation_rows_csv)
    if public_evaluation_rows_csv is not None and not scorecard.public_evaluation_rows.empty:
        public_evaluation_rows_csv = Path(public_evaluation_rows_csv)
        public_evaluation_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard.public_evaluation_rows.to_csv(public_evaluation_rows_csv, index=False)
        paths["public_evaluation_rows_csv"] = str(public_evaluation_rows_csv)
    if nearest_time_rows_csv is not None and not scorecard.nearest_time_rows.empty:
        nearest_time_rows_csv = Path(nearest_time_rows_csv)
        nearest_time_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard.nearest_time_rows.to_csv(nearest_time_rows_csv, index=False)
        paths["nearest_time_rows_csv"] = str(nearest_time_rows_csv)
    return paths


def scorecard_summary_frame(summary: dict[str, Any]) -> pd.DataFrame:
    """Flatten the main scorecard fields into a one-row table."""

    validation = summary.get("validation", {})
    public = summary.get("public_track5", {})
    public_pooled = public.get("pooled", {}) if isinstance(public, dict) else {}
    nearest = summary.get("nearest_time", {})
    nearest_pooled = nearest.get("pooled", {}) if isinstance(nearest, dict) else {}
    row = {
        "results_path": summary.get("results_path"),
        "truth_path": summary.get("truth_path"),
        "template_path": summary.get("template_path"),
        "upload_manifest_path": summary.get("upload_manifest_path"),
        "codabench_upload_ready": validation.get("codabench_upload_ready"),
        "upload_manifest_valid": summary.get("upload_manifest_valid"),
        "upload_manifest_codabench_upload_ready": summary.get(
            "upload_manifest_codabench_upload_ready"
        ),
        "validation_leaderboard_ready": validation.get("leaderboard_ready"),
        "public_metric_leaderboard_ready": public.get("leaderboard_ready")
        if isinstance(public, dict)
        else None,
        "scorecard_leaderboard_ready": summary.get("scorecard_leaderboard_ready"),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in summary.get("leaderboard_blocking_reasons", [])
        ),
        "pose_mse_loss_m2": public_pooled.get("pose_mse_loss_m2"),
        "public_mean_3d_m": public_pooled.get("mean_3d_m"),
        "public_rmse_3d_m": public_pooled.get("rmse_3d_m"),
        "public_p95_3d_m": public_pooled.get("p95_3d_m"),
        "public_max_3d_m": public_pooled.get("max_3d_m"),
        "uav_type_accuracy": public_pooled.get("uav_type_accuracy"),
        "nearest_mean_3d_m": nearest_pooled.get("mean_3d_m"),
        "nearest_p95_3d_m": nearest_pooled.get("p95_3d_m"),
        "nearest_max_3d_m": nearest_pooled.get("max_3d_m"),
        "truth_count": public.get("truth_count") if isinstance(public, dict) else None,
        "prediction_count": public.get("prediction_count") if isinstance(public, dict) else None,
        "matched_count": public.get("matched_count") if isinstance(public, dict) else None,
        "missing_prediction_count": public.get("missing_prediction_count")
        if isinstance(public, dict)
        else None,
        "extra_prediction_count": public.get("extra_prediction_count")
        if isinstance(public, dict)
        else None,
        "duplicate_prediction_count": public.get("duplicate_prediction_count")
        if isinstance(public, dict)
        else None,
    }
    return pd.DataFrame([row])


def _scorecard_template_frame(*, truth_frame, template_path: Path | None) -> pd.DataFrame | None:
    if template_path is not None:
        return load_official_track5_template_file(template_path)
    if truth_frame is None:
        return None
    rows = truth_frame.rows[["sequence_id", "time_s"]].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    return rows.loc[rows["time_s"].notna()].drop_duplicates().reset_index(drop=True)


def _scorecard_summary(
    *,
    results_path: Path,
    truth_path: Path | None,
    template_path: Path | None,
    class_map_path: Path | None,
    upload_manifest_path: Path | None,
    require_zip: bool,
    timestamp_tolerance_s: float,
    max_time_delta_s: float,
    validation: OfficialTrack5Validation,
    public_eval: dict[str, Any] | None,
    nearest_eval: dict[str, Any] | None,
    manifest_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    public_summary = public_eval["summary"] if public_eval is not None else None
    nearest_summary = nearest_eval["summary"] if nearest_eval is not None else None
    reasons = list(validation.summary.get("leaderboard_blocking_reasons") or [])
    if public_summary is None:
        reasons.append("public_track5_evaluation_not_run")
        public_ready = False
    else:
        public_ready = bool(public_summary.get("leaderboard_ready", False))
        reasons.extend(str(item) for item in public_summary.get("leaderboard_blocking_reasons", []))
    if manifest_verification is None:
        manifest_ready = True
    else:
        manifest_ready = bool(manifest_verification.get("codabench_upload_ready", False))
        if not manifest_verification.get("valid", False):
            reasons.append("official_upload_manifest_invalid")
        if not manifest_verification.get("codabench_upload_ready", False):
            reasons.append("official_upload_manifest_not_upload_ready")
    unique_reasons = sorted(set(str(reason) for reason in reasons if str(reason)))
    return {
        "schema": "raft-uav-mmuad-track5-scorecard-v1",
        "closed_codabench_evaluator": False,
        "description": (
            "Local preflight scorecard combining official-style upload validation, "
            "timestamp-aligned Track 5 metrics, and nearest-time diagnostics."
        ),
        "results_path": str(results_path),
        "truth_path": str(truth_path) if truth_path is not None else None,
        "template_path": str(template_path) if template_path is not None else None,
        "class_map_path": str(class_map_path) if class_map_path is not None else None,
        "upload_manifest_path": (
            str(upload_manifest_path) if upload_manifest_path is not None else None
        ),
        "require_zip": bool(require_zip),
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
        "max_time_delta_s": float(max_time_delta_s),
        "validation": validation.summary,
        "upload_manifest_verification": manifest_verification,
        "upload_manifest_valid": (
            bool(manifest_verification.get("valid", False))
            if manifest_verification is not None
            else None
        ),
        "upload_manifest_codabench_upload_ready": (
            bool(manifest_verification.get("codabench_upload_ready", False))
            if manifest_verification is not None
            else None
        ),
        "public_track5": public_summary,
        "nearest_time": nearest_summary,
        "scorecard_leaderboard_ready": bool(
            validation.summary.get("leaderboard_ready", False)
            and public_ready
            and manifest_ready
        ),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "leaderboard_blocking_reasons": unique_reasons,
    }
