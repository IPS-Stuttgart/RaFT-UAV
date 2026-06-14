"""Local leaderboard aggregation for MMUAD/UG2-style result files.

This module intentionally builds repository-local evidence tables.  It does not
claim closed Codabench equivalence; it reuses the transparent local evaluator
implemented in :mod:`raft_uav.mmuad.evaluator` and records the protocol in every
row.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_file,
)
from raft_uav.mmuad.schema import load_jsonable


DEFAULT_RANK_METRIC = "pose_mse_loss_m2"


@dataclass(frozen=True)
class LeaderboardEntry:
    """One method/result file to evaluate for a local leaderboard."""

    method: str
    results_path: Path
    truth_path: Path
    metric_protocol: str = "public-track5"
    source_note: str = ""
    class_map_path: Path | None = None
    max_time_delta_s: float = 0.5
    timestamp_tolerance_s: float = 1.0e-6


@dataclass(frozen=True)
class LeaderboardResult:
    """Evaluated leaderboard rows and full evaluator payloads."""

    rows: pd.DataFrame
    evaluations: dict[str, dict[str, Any]]


def load_leaderboard_config(path: Path) -> list[LeaderboardEntry]:
    """Load a local leaderboard configuration from JSON/YAML or CSV.

    JSON/YAML example::

        {
          "default_truth": "truth.csv",
          "default_metric_protocol": "public-track5",
          "methods": [
            {"method": "baseline", "results": "baseline.csv"},
            {"name": "ours", "results_csv": "ours.zip"}
          ]
        }

    CSV example::

        method,results_csv,truth_csv,metric_protocol,source_note
        baseline,baseline.csv,truth.csv,public-track5,
    """

    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        payload: dict[str, Any] = {"methods": frame.to_dict(orient="records")}
    else:
        payload = _load_mapping_file(path)
    return leaderboard_entries_from_config(payload, base_dir=path.parent)


def leaderboard_entries_from_config(
    payload: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> list[LeaderboardEntry]:
    """Return normalized leaderboard entries from a config mapping."""

    base = Path(".") if base_dir is None else Path(base_dir)
    raw_methods = payload.get("methods", payload.get("entries", payload.get("rows", [])))
    if not isinstance(raw_methods, list):
        raise ValueError("leaderboard config must contain a list under methods/entries/rows")
    default_truth = payload.get("default_truth", payload.get("truth", payload.get("truth_csv")))
    default_protocol = str(payload.get("default_metric_protocol", "public-track5"))
    default_class_map = payload.get("default_class_map", payload.get("class_map"))
    default_time_delta = float(payload.get("default_max_time_delta_s", 0.5))
    default_tolerance = float(payload.get("default_timestamp_tolerance_s", 1.0e-6))

    entries: list[LeaderboardEntry] = []
    for index, raw in enumerate(raw_methods):
        if not isinstance(raw, dict):
            raise ValueError(f"leaderboard method entry {index} must be an object")
        method = str(raw.get("method", raw.get("name", raw.get("label", f"method_{index}"))))
        result_value = raw.get("results", raw.get("results_csv", raw.get("results_zip")))
        if result_value is None:
            raise ValueError(f"leaderboard entry {method!r} is missing results/results_csv")
        truth_value = raw.get("truth", raw.get("truth_csv", raw.get("truth_file", default_truth)))
        if truth_value is None:
            raise ValueError(f"leaderboard entry {method!r} is missing truth/default_truth")
        class_map_value = raw.get("class_map", raw.get("class_map_csv", default_class_map))
        entries.append(
            LeaderboardEntry(
                method=method,
                results_path=_resolve_config_path(base, result_value),
                truth_path=_resolve_config_path(base, truth_value),
                metric_protocol=str(raw.get("metric_protocol", default_protocol)),
                source_note=str(raw.get("source_note", raw.get("note", ""))),
                class_map_path=(
                    _resolve_config_path(base, class_map_value)
                    if class_map_value not in {None, ""}
                    else None
                ),
                max_time_delta_s=float(raw.get("max_time_delta_s", default_time_delta)),
                timestamp_tolerance_s=float(
                    raw.get("timestamp_tolerance_s", default_tolerance)
                ),
            )
        )
    if not entries:
        raise ValueError("leaderboard config contains no method entries")
    return entries


def build_mmuad_leaderboard(
    entries: Iterable[LeaderboardEntry],
    *,
    rank_metric: str = DEFAULT_RANK_METRIC,
) -> LeaderboardResult:
    """Evaluate result files and return a ranked local leaderboard."""

    rows: list[dict[str, Any]] = []
    evaluations: dict[str, dict[str, Any]] = {}
    for entry in entries:
        evaluation = evaluate_mmaud_results(
            load_mmaud_results_file(entry.results_path),
            load_evaluation_truth_file(entry.truth_path),
            max_time_delta_s=entry.max_time_delta_s,
            metric_protocol=entry.metric_protocol,
            timestamp_tolerance_s=entry.timestamp_tolerance_s,
            class_map_path=entry.class_map_path,
        )
        evaluations[entry.method] = evaluation["summary"]
        rows.append(_leaderboard_row(entry, evaluation["summary"]))
    frame = pd.DataFrame.from_records(rows)
    frame = rank_leaderboard_frame(frame, rank_metric=rank_metric)
    return LeaderboardResult(rows=frame, evaluations=evaluations)


def rank_leaderboard_frame(
    frame: pd.DataFrame,
    *,
    rank_metric: str = DEFAULT_RANK_METRIC,
) -> pd.DataFrame:
    """Sort a leaderboard frame and add a one-based rank column."""

    if frame.empty:
        return frame.assign(rank=[])
    work = frame.copy()
    metric = rank_metric if rank_metric in work.columns else _fallback_rank_metric(work)
    sort_columns = [metric]
    ascending = [True]
    for candidate, asc in (
        ("p95_3d_m", True),
        ("max_3d_m", True),
        ("uav_type_accuracy", False),
        ("method", True),
    ):
        if candidate in work.columns and candidate != metric:
            sort_columns.append(candidate)
            ascending.append(asc)
    work = work.sort_values(sort_columns, ascending=ascending, na_position="last")
    work.insert(0, "rank", range(1, len(work) + 1))
    work["rank_metric"] = metric
    return work.reset_index(drop=True)


def write_leaderboard_artifacts(
    result: LeaderboardResult,
    *,
    output_dir: Path,
    stem: str = "mmuad_local_leaderboard",
) -> dict[str, str]:
    """Write CSV/JSON/Markdown leaderboard artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{stem}.csv"
    json_path = output / f"{stem}.json"
    md_path = output / f"{stem}.md"
    result.rows.to_csv(csv_path, index=False)
    json_payload = {
        "schema": "raft-uav-mmuad-local-leaderboard-v1",
        "rows": result.rows.to_dict(orient="records"),
        "evaluations": result.evaluations,
    }
    json_path.write_text(json.dumps(load_jsonable(json_payload), indent=2), encoding="utf-8")
    md_path.write_text(_leaderboard_markdown(result.rows), encoding="utf-8")
    return {
        "leaderboard_csv": str(csv_path),
        "leaderboard_json": str(json_path),
        "leaderboard_md": str(md_path),
    }


def _leaderboard_row(entry: LeaderboardEntry, summary: dict[str, Any]) -> dict[str, Any]:
    pooled = summary.get("pooled", {}) if isinstance(summary, dict) else {}
    return {
        "method": entry.method,
        "metric_protocol": summary.get("metric_protocol", entry.metric_protocol),
        "public_track5_metric": summary.get("public_track5_metric", False),
        "leaderboard_ready": summary.get("leaderboard_ready", False),
        "score_valid_for_leaderboard": summary.get("score_valid_for_leaderboard", False),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in summary.get("leaderboard_blocking_reasons", [])
        ),
        "truth_count": summary.get("truth_count", pooled.get("truth_count")),
        "prediction_count": summary.get("prediction_count", pooled.get("prediction_count")),
        "matched_count": summary.get("matched_count", pooled.get("matched_count")),
        "truth_coverage_fraction": summary.get("truth_coverage_fraction"),
        "pose_mse_loss_m2": pooled.get("pose_mse_loss_m2"),
        "mean_square_loss_m2": pooled.get("mean_square_loss_m2"),
        "mean_3d_m": pooled.get("mean_3d_m"),
        "rmse_3d_m": pooled.get("rmse_3d_m"),
        "p50_3d_m": pooled.get("p50_3d_m"),
        "p95_3d_m": pooled.get("p95_3d_m"),
        "max_3d_m": pooled.get("max_3d_m"),
        "mean_2d_m": pooled.get("mean_2d_m"),
        "p95_2d_m": pooled.get("p95_2d_m"),
        "max_2d_m": pooled.get("max_2d_m"),
        "uav_type_count": pooled.get("uav_type_count"),
        "uav_type_accuracy": pooled.get("uav_type_accuracy"),
        "classification_accuracy": pooled.get("classification_accuracy"),
        "source_note": entry.source_note,
        "results_path": str(entry.results_path),
        "truth_path": str(entry.truth_path),
        "class_map_path": str(entry.class_map_path) if entry.class_map_path else "",
    }


def _leaderboard_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# MMUAD local leaderboard\n\nNo rows.\n"
    columns = [
        column
        for column in (
            "rank",
            "method",
            "metric_protocol",
            "pose_mse_loss_m2",
            "mean_3d_m",
            "p95_3d_m",
            "max_3d_m",
            "uav_type_accuracy",
            "leaderboard_ready",
            "source_note",
        )
        if column in frame.columns
    ]
    lines = [
        "# MMUAD local leaderboard",
        "",
        "This is a repository-local leaderboard built with RaFT-UAV's transparent evaluator; it is not the closed Codabench runtime.",
        "",
        frame[columns].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def _fallback_rank_metric(frame: pd.DataFrame) -> str:
    for candidate in ("mean_3d_m", "rmse_3d_m", "max_3d_m"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("leaderboard rows do not contain a supported rank metric")


def _load_mapping_file(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception:
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("leaderboard config must be a mapping/object")
    return payload


def _resolve_config_path(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path
