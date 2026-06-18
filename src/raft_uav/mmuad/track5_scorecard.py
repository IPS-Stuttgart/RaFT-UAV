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

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_file,
)
from raft_uav.mmuad.schema import load_jsonable
from raft_uav.mmuad.sequence import (
    OFFICIAL_TRACK5_TIMESTAMP_SOURCES,
    discover_sequence_paths,
    official_track5_timestamp_template,
)
from raft_uav.mmuad.splits import filter_sequences_by_split_folder
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
    pose_by_sequence: pd.DataFrame
    candidate_regret_summary: pd.DataFrame


def build_track5_scorecard(
    *,
    results_path: Path,
    truth_path: Path | None = None,
    template_path: Path | None = None,
    sequence_root: Path | None = None,
    sequence_glob: str = "*",
    split_name: str | None = None,
    timestamp_source: str = "ground-truth-or-all",
    class_map_path: Path | None = None,
    upload_manifest_path: Path | None = None,
    classification_provenance_path: Path | None = None,
    selected_tracklets_path: Path | None = None,
    candidate_oracle_gap_path: Path | None = None,
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
    sequence_root:
        Optional public Track 5 sequence root used to build the timestamp
        template directly from folders such as ``Image`` or ``ground_truth``.
        This closes the previous workflow gap where users first had to generate
        a separate template file before running the scorecard.
    sequence_glob:
        Glob passed to sequence discovery when ``sequence_root`` is supplied.
    split_name:
        Optional top-level split folder name, for layouts such as ``val/seq001``.
    timestamp_source:
        Public Track 5 modality folder used for the template when
        ``sequence_root`` is supplied.
    class_map_path:
        Optional sequence-to-class map for evaluation.
    upload_manifest_path:
        Optional ``mmuad_official_upload_manifest.json`` written by the
        official validation path.  When supplied, scorecard readiness also
        requires the manifest fingerprints to match the current artifact.
    classification_provenance_path:
        Optional classifier provenance JSON written by ``raft-uav-mmuad-run``
        when ``--sequence-classifier`` is used.
    selected_tracklets_path:
        Optional ``mmuad_selected_tracklets.csv`` from a tracker run.  When
        supplied, the paper pose-by-sequence table includes selected-source
        dominance/count fields.
    candidate_oracle_gap_path:
        Optional ``mmuad_candidate_oracle_gap.csv``.  When supplied, the
        scorecard writes candidate-regret summary rows and can report
        radar-empty counts in the pose-by-sequence table.
    require_zip:
        Whether validation should require a ZIP.  Keep this true for Codabench
        preflight; set false for local CSV development checks.
    """

    if template_path is not None and sequence_root is not None:
        raise ValueError("pass either template_path or sequence_root, not both")
    if timestamp_source not in OFFICIAL_TRACK5_TIMESTAMP_SOURCES:
        allowed = ", ".join(OFFICIAL_TRACK5_TIMESTAMP_SOURCES)
        raise ValueError(f"unsupported timestamp_source {timestamp_source!r}; allowed={allowed}")

    results_path = Path(results_path)
    truth_path = Path(truth_path) if truth_path is not None else None
    template_path = Path(template_path) if template_path is not None else None
    sequence_root = Path(sequence_root) if sequence_root is not None else None
    class_map_path = Path(class_map_path) if class_map_path is not None else None
    upload_manifest_path = (
        Path(upload_manifest_path) if upload_manifest_path is not None else None
    )
    selected_tracklets_path = (
        Path(selected_tracklets_path) if selected_tracklets_path is not None else None
    )
    candidate_oracle_gap_path = (
        Path(candidate_oracle_gap_path) if candidate_oracle_gap_path is not None else None
    )
    classification_provenance_path = (
        Path(classification_provenance_path)
        if classification_provenance_path is not None
        else None
    )
    classification_provenance = _load_classification_provenance(
        classification_provenance_path
    )

    truth_frame = load_evaluation_truth_file(truth_path) if truth_path is not None else None
    template = _scorecard_template_frame(
        truth_frame=truth_frame,
        template_path=template_path,
        sequence_root=sequence_root,
        sequence_glob=sequence_glob,
        split_name=split_name,
        timestamp_source=timestamp_source,
    )
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
    selected_tracklets = _load_optional_csv(selected_tracklets_path)
    candidate_oracle_gap = _load_optional_csv(candidate_oracle_gap_path)
    candidate_regret_summary = build_candidate_regret_summary(candidate_oracle_gap)
    pose_by_sequence = build_pose_by_sequence_table(
        public_rows,
        selected_tracklets=selected_tracklets,
        candidate_oracle_gap=candidate_oracle_gap,
    )

    summary = _scorecard_summary(
        results_path=results_path,
        truth_path=truth_path,
        template_path=template_path,
        sequence_root=sequence_root,
        sequence_glob=sequence_glob,
        split_name=split_name,
        timestamp_source=timestamp_source,
        class_map_path=class_map_path,
        upload_manifest_path=upload_manifest_path,
        selected_tracklets_path=selected_tracklets_path,
        candidate_oracle_gap_path=candidate_oracle_gap_path,
        classification_provenance=classification_provenance,
        require_zip=require_zip,
        timestamp_tolerance_s=timestamp_tolerance_s,
        max_time_delta_s=max_time_delta_s,
        validation=validation,
        public_eval=public_eval,
        nearest_eval=nearest_eval,
        manifest_verification=manifest_verification,
        pose_by_sequence=pose_by_sequence,
        candidate_regret_summary=candidate_regret_summary,
    )
    return Track5Scorecard(
        summary=summary,
        validation_rows=validation.rows,
        public_evaluation_rows=public_rows,
        nearest_time_rows=nearest_rows,
        pose_by_sequence=pose_by_sequence,
        candidate_regret_summary=candidate_regret_summary,
    )


def write_track5_scorecard(
    scorecard: Track5Scorecard,
    *,
    summary_json: Path,
    summary_csv: Path | None = None,
    validation_rows_csv: Path | None = None,
    public_evaluation_rows_csv: Path | None = None,
    nearest_time_rows_csv: Path | None = None,
    pose_by_sequence_csv: Path | None = None,
    candidate_regret_summary_csv: Path | None = None,
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
    if pose_by_sequence_csv is not None and not scorecard.pose_by_sequence.empty:
        pose_by_sequence_csv = Path(pose_by_sequence_csv)
        pose_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard.pose_by_sequence.to_csv(pose_by_sequence_csv, index=False)
        paths["pose_by_sequence_csv"] = str(pose_by_sequence_csv)
    if (
        candidate_regret_summary_csv is not None
        and not scorecard.candidate_regret_summary.empty
    ):
        candidate_regret_summary_csv = Path(candidate_regret_summary_csv)
        candidate_regret_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        scorecard.candidate_regret_summary.to_csv(candidate_regret_summary_csv, index=False)
        paths["candidate_regret_summary_csv"] = str(candidate_regret_summary_csv)
    return paths


def build_pose_by_sequence_table(
    public_evaluation_rows: pd.DataFrame,
    *,
    selected_tracklets: pd.DataFrame | None = None,
    candidate_oracle_gap: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return paper-facing per-sequence pose diagnostics."""

    rows = pd.DataFrame(public_evaluation_rows).copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return pd.DataFrame(columns=_pose_by_sequence_columns())
    matched_mask = _bool_series(rows["matched"]) if "matched" in rows.columns else pd.Series(
        False,
        index=rows.index,
    )
    matched = rows.loc[matched_mask.to_numpy(bool)].copy()
    if matched.empty:
        return pd.DataFrame(columns=_pose_by_sequence_columns())
    for column in ("error_3d_m", "squared_error_3d_m2"):
        matched[column] = _numeric_column(matched, column)
    selected_stats = _selected_sensor_stats(selected_tracklets)
    radar_empty_counts = _empty_radar_counts(candidate_oracle_gap)
    records: list[dict[str, Any]] = []
    for sequence_id, group in matched.groupby(matched["sequence_id"].astype(str), sort=True):
        errors = pd.to_numeric(group["error_3d_m"], errors="coerce").dropna()
        squared = pd.to_numeric(group["squared_error_3d_m2"], errors="coerce").dropna()
        stats = selected_stats.get(str(sequence_id), {})
        mse = float(squared.mean()) if not squared.empty else None
        records.append(
            {
                "sequence": str(sequence_id),
                "sequence_id": str(sequence_id),
                "count": int(len(errors)),
                "mse": mse,
                "rmse": (float(mse**0.5) if mse is not None else None),
                "mean_3d": _series_mean(errors),
                "median_3d": _series_quantile(errors, 0.5),
                "p95_3d": _series_quantile(errors, 0.95),
                "max_3d": _series_max(errors),
                "dominant_sensor": stats.get("dominant_sensor", "unknown"),
                "used_lidar_360_count": int(stats.get("used_lidar_360_count", 0)),
                "used_livox_avia_count": int(stats.get("used_livox_avia_count", 0)),
                "used_radar_count": int(stats.get("used_radar_count", 0)),
                "empty_radar_count": radar_empty_counts.get(str(sequence_id)),
            }
        )
    return pd.DataFrame.from_records(records, columns=_pose_by_sequence_columns())


def build_candidate_regret_summary(candidate_oracle_gap: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize ``mmuad_candidate_oracle_gap.csv`` for paper artifacts."""

    rows = pd.DataFrame() if candidate_oracle_gap is None else pd.DataFrame(candidate_oracle_gap).copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return pd.DataFrame(columns=_candidate_regret_summary_columns())
    if "sensor" not in rows.columns:
        rows["sensor"] = "candidate"
    for column in (
        "candidate_regret_m",
        "selected_minus_truth_error_m",
        "nearest_minus_truth_error_m",
        "nearest_candidate_time_delta_s",
        "candidate_count_at_nearest_time",
    ):
        rows[column] = _numeric_column(rows, column)
    records: list[dict[str, Any]] = []
    for (sequence_id, sensor), group in rows.groupby(
        [rows["sequence_id"].astype(str), rows["sensor"].astype(str)],
        sort=True,
    ):
        regret = pd.to_numeric(group["candidate_regret_m"], errors="coerce").dropna()
        selected_error = pd.to_numeric(
            group["selected_minus_truth_error_m"], errors="coerce"
        ).dropna()
        nearest_error = pd.to_numeric(
            group["nearest_minus_truth_error_m"], errors="coerce"
        ).dropna()
        nearest_found = _bool_series(group.get("nearest_candidate_found"))
        selected_found = _bool_series(group.get("selected_candidate_found"))
        source_match = _bool_series(group.get("selected_source_matches_sensor"))
        records.append(
            {
                "sequence": str(sequence_id),
                "sequence_id": str(sequence_id),
                "sensor": str(sensor),
                "row_count": int(len(group)),
                "nearest_found_fraction": _bool_mean(nearest_found),
                "selected_found_fraction": _bool_mean(selected_found),
                "selected_source_match_fraction": _bool_mean(source_match),
                "empty_candidate_frame_count": int(
                    (group["candidate_count_at_nearest_time"].fillna(0.0) <= 0.0).sum()
                ),
                "mean_selected_error_m": _series_mean(selected_error),
                "p95_selected_error_m": _series_quantile(selected_error, 0.95),
                "mean_nearest_error_m": _series_mean(nearest_error),
                "p95_nearest_error_m": _series_quantile(nearest_error, 0.95),
                "mean_candidate_regret_m": _series_mean(regret),
                "median_candidate_regret_m": _series_quantile(regret, 0.5),
                "p95_candidate_regret_m": _series_quantile(regret, 0.95),
                "max_candidate_regret_m": _series_max(regret),
                "positive_regret_fraction": _positive_fraction(regret),
                "p95_abs_nearest_time_delta_s": _series_quantile(
                    group["nearest_candidate_time_delta_s"].abs().dropna(),
                    0.95,
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=_candidate_regret_summary_columns())


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
        "sequence_root": summary.get("sequence_root"),
        "sequence_glob": summary.get("sequence_glob"),
        "split_name": summary.get("split_name"),
        "timestamp_source": summary.get("timestamp_source"),
        "upload_manifest_path": summary.get("upload_manifest_path"),
        "selected_tracklets_path": summary.get("selected_tracklets_path"),
        "candidate_oracle_gap_path": summary.get("candidate_oracle_gap_path"),
        "classification_model_path": summary.get("classification_model_path"),
        "classification_method": summary.get("classification_method"),
        "classification_train_sequences": ";".join(
            str(item) for item in summary.get("classification_train_sequences", []) or []
        ),
        "classification_feature_columns": ";".join(
            str(item) for item in summary.get("classification_feature_columns", []) or []
        ),
        "classification_class_map": json.dumps(
            summary.get("classification_class_map", {}) or {},
            sort_keys=True,
        ),
        "classification_prediction_mode": summary.get("classification_prediction_mode"),
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
        "classification_accuracy": public_pooled.get("classification_accuracy"),
        "nearest_mean_3d_m": nearest_pooled.get("mean_3d_m"),
        "nearest_p95_3d_m": nearest_pooled.get("p95_3d_m"),
        "nearest_max_3d_m": nearest_pooled.get("max_3d_m"),
        "paper_pose_by_sequence_rows": (
            summary.get("paper_artifacts", {}) or {}
        ).get("pose_by_sequence_rows"),
        "paper_candidate_regret_summary_rows": (
            summary.get("paper_artifacts", {}) or {}
        ).get("candidate_regret_summary_rows"),
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


def _pose_by_sequence_columns() -> list[str]:
    return [
        "sequence",
        "sequence_id",
        "count",
        "mse",
        "rmse",
        "mean_3d",
        "median_3d",
        "p95_3d",
        "max_3d",
        "dominant_sensor",
        "used_lidar_360_count",
        "used_livox_avia_count",
        "used_radar_count",
        "empty_radar_count",
    ]


def _candidate_regret_summary_columns() -> list[str]:
    return [
        "sequence",
        "sequence_id",
        "sensor",
        "row_count",
        "nearest_found_fraction",
        "selected_found_fraction",
        "selected_source_match_fraction",
        "empty_candidate_frame_count",
        "mean_selected_error_m",
        "p95_selected_error_m",
        "mean_nearest_error_m",
        "p95_nearest_error_m",
        "mean_candidate_regret_m",
        "median_candidate_regret_m",
        "p95_candidate_regret_m",
        "max_candidate_regret_m",
        "positive_regret_fraction",
        "p95_abs_nearest_time_delta_s",
    ]


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)


def _selected_sensor_stats(selected_tracklets: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if selected_tracklets is None:
        return {}
    rows = pd.DataFrame(selected_tracklets).copy()
    if rows.empty or "sequence_id" not in rows.columns or "source" not in rows.columns:
        return {}
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_sensor_bucket"] = rows["source"].map(_sensor_bucket)
    stats: dict[str, dict[str, Any]] = {}
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        counts = group["_sensor_bucket"].value_counts()
        dominant = str(counts.index[0]) if not counts.empty else "unknown"
        stats[str(sequence_id)] = {
            "dominant_sensor": dominant,
            "used_lidar_360_count": int(counts.get("lidar_360", 0)),
            "used_livox_avia_count": int(counts.get("livox_avia", 0)),
            "used_radar_count": int(counts.get("radar", 0)),
        }
    return stats


def _empty_radar_counts(candidate_oracle_gap: pd.DataFrame | None) -> dict[str, int | None]:
    if candidate_oracle_gap is None:
        return {}
    rows = pd.DataFrame(candidate_oracle_gap).copy()
    if rows.empty or "sequence_id" not in rows.columns or "sensor" not in rows.columns:
        return {}
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_sensor_bucket"] = rows["sensor"].map(_sensor_bucket)
    rows["candidate_count_at_nearest_time"] = pd.to_numeric(
        rows.get("candidate_count_at_nearest_time"),
        errors="coerce",
    )
    radar = rows.loc[rows["_sensor_bucket"] == "radar"].copy()
    if radar.empty:
        return {}
    empty = radar.loc[radar["candidate_count_at_nearest_time"].fillna(0.0) <= 0.0]
    counts = empty.groupby("sequence_id", sort=True).size()
    sequences = sorted(radar["sequence_id"].unique())
    return {sequence: int(counts.get(sequence, 0)) for sequence in sequences}


def _sensor_bucket(value: object) -> str:
    text = "" if value is None else str(value)
    norm = text.strip().lower().replace("-", "_").replace(" ", "_")
    if "radar" in norm:
        return "radar"
    if "livox" in norm or "avia" in norm:
        return "livox_avia"
    if "lidar" in norm:
        return "lidar_360"
    return norm or "unknown"


def _bool_series(values: Any) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    series = pd.Series(values)
    if series.empty:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series.astype(bool)
    text = series.fillna(False).astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _bool_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.astype(bool).mean())


def _series_mean(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(float(quantile))) if not values.empty else None


def _series_max(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if not values.empty else None


def _positive_fraction(values: pd.Series) -> float | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0.0).mean())


def template_frame_from_sequence_root(
    sequence_root: Path,
    *,
    sequence_glob: str = "*",
    split_name: str | None = None,
    timestamp_source: str = "ground-truth-or-all",
) -> pd.DataFrame:
    """Build a Track 5 timestamp template from a public sequence root.

    The returned frame has the normalized ``sequence_id,time_s`` columns accepted
    by the existing official-submission validator.  It is intentionally a
    timestamp template only; hidden validation/test labels are not inferred.
    """

    if timestamp_source not in OFFICIAL_TRACK5_TIMESTAMP_SOURCES:
        allowed = ", ".join(OFFICIAL_TRACK5_TIMESTAMP_SOURCES)
        raise ValueError(f"unsupported timestamp_source {timestamp_source!r}; allowed={allowed}")
    root = Path(sequence_root)
    sequences = discover_sequence_paths(root, sequence_glob=sequence_glob)
    if split_name is not None:
        sequences = filter_sequences_by_split_folder(sequences, root, split_name)
    if not sequences:
        split_suffix = f" for split {split_name!r}" if split_name is not None else ""
        raise ValueError(f"no Track 5 sequences discovered in {root}{split_suffix}")

    frames: list[pd.DataFrame] = []
    for sequence in sequences:
        template = official_track5_timestamp_template(
            sequence,
            timestamp_source=timestamp_source,
        )
        if not template.rows.empty:
            frames.append(template.rows[["sequence_id", "time_s"]].copy())
    if not frames:
        raise ValueError(
            f"no Track 5 timestamps discovered in {root} using source {timestamp_source!r}"
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="coerce")
    frame = frame.loc[frame["time_s"].notna()].drop_duplicates()
    return frame.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _scorecard_template_frame(
    *,
    truth_frame,
    template_path: Path | None,
    sequence_root: Path | None,
    sequence_glob: str,
    split_name: str | None,
    timestamp_source: str,
) -> pd.DataFrame | None:
    if template_path is not None:
        return load_official_track5_template_file(template_path)
    if sequence_root is not None:
        return template_frame_from_sequence_root(
            sequence_root,
            sequence_glob=sequence_glob,
            split_name=split_name,
            timestamp_source=timestamp_source,
        )
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
    sequence_root: Path | None,
    sequence_glob: str,
    split_name: str | None,
    timestamp_source: str,
    class_map_path: Path | None,
    upload_manifest_path: Path | None,
    selected_tracklets_path: Path | None,
    candidate_oracle_gap_path: Path | None,
    require_zip: bool,
    timestamp_tolerance_s: float,
    max_time_delta_s: float,
    validation: OfficialTrack5Validation,
    public_eval: dict[str, Any] | None,
    nearest_eval: dict[str, Any] | None,
    manifest_verification: dict[str, Any] | None,
    pose_by_sequence: pd.DataFrame,
    candidate_regret_summary: pd.DataFrame,
    classification_provenance: dict[str, Any] | None,
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
    summary = {
        "schema": "raft-uav-mmuad-track5-scorecard-v1",
        "closed_codabench_evaluator": False,
        "description": (
            "Local preflight scorecard combining official-style upload validation, "
            "timestamp-aligned Track 5 metrics, and nearest-time diagnostics."
        ),
        "results_path": str(results_path),
        "truth_path": str(truth_path) if truth_path is not None else None,
        "template_path": str(template_path) if template_path is not None else None,
        "sequence_root": str(sequence_root) if sequence_root is not None else None,
        "sequence_glob": str(sequence_glob),
        "split_name": str(split_name) if split_name is not None else None,
        "timestamp_source": str(timestamp_source),
        "class_map_path": str(class_map_path) if class_map_path is not None else None,
        "upload_manifest_path": (
            str(upload_manifest_path) if upload_manifest_path is not None else None
        ),
        "selected_tracklets_path": (
            str(selected_tracklets_path) if selected_tracklets_path is not None else None
        ),
        "candidate_oracle_gap_path": (
            str(candidate_oracle_gap_path) if candidate_oracle_gap_path is not None else None
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
        "paper_artifacts": {
            "pose_by_sequence_rows": int(len(pose_by_sequence)),
            "candidate_regret_summary_rows": int(len(candidate_regret_summary)),
            "selected_tracklets_available": selected_tracklets_path is not None,
            "candidate_oracle_gap_available": candidate_oracle_gap_path is not None,
        },
    }
    summary.update(
        _classification_provenance_fields(
            classification_provenance,
            manifest_verification=manifest_verification,
        )
    )
    return summary


def _load_classification_provenance(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("classification provenance JSON must contain an object")
    return payload


def _classification_provenance_fields(
    provenance: dict[str, Any] | None,
    *,
    manifest_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    source = provenance or manifest_verification or {}
    keys = (
        "classification_model_path",
        "classification_method",
        "classification_train_sequences",
        "classification_feature_columns",
        "classification_class_map",
        "classification_prediction_mode",
    )
    return {key: source.get(key) for key in keys}
