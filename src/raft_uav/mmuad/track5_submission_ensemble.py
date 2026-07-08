"""Ensemble official MMUAD/UG2+ Track 5 submissions.

Leaderboard experiments often produce several upload-ready ``mmaud_results.csv``
files from different candidate reservoirs, calibration branches, or mixture-MAP
settings.  This module averages their 3D positions on the official
``Sequence,Timestamp`` grid and combines classification with a weighted vote.
It is inference-safe: it uses only submitted predictions and an optional official
template for validation, never truth positions.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_context import _predicted_class_labels
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    normalize_official_track5_results_frame,
    parse_official_position_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

RESULTS_CSV_NAME = "mmaud_results.csv"
ENSEMBLED_ESTIMATES_CSV = "mmuad_track5_ensemble_estimates.csv"
ENSEMBLED_RESULTS_CSV = "mmaud_results_ensemble.csv"
ENSEMBLED_ZIP = "ug2_submission_ensemble.zip"
ENSEMBLE_DIAGNOSTICS_CSV = "mmuad_track5_ensemble_diagnostics.csv"
ENSEMBLE_MANIFEST_JSON = "mmuad_track5_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_ensemble_validation_rows.csv"
SEQUENCE_COLUMN = "Sequence"
TIME_COLUMN = "Timestamp"
POSITION_COLUMN = "Position"
CLASSIFICATION_COLUMN = "Classification"
_NORMALIZED_REQUIRED_COLUMNS = ("sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m")
_NORMALIZED_CLASSIFICATION_ALIASES = ("Classification", "classification")


@dataclass(frozen=True)
class SubmissionInput:
    """Weighted Track 5 submission input."""

    label: str
    path: Path
    weight: float = 1.0


def parse_submission_input(value: str) -> SubmissionInput:
    """Parse ``LABEL=WEIGHT:PATH``, ``WEIGHT:PATH``, or plain path specs."""

    label = ""
    rest = str(value)
    if "=" in rest:
        label, rest = rest.split("=", 1)
    weight = 1.0
    path_text = rest
    if ":" in rest:
        maybe_weight, maybe_path = rest.split(":", 1)
        try:
            weight = float(maybe_weight)
            path_text = maybe_path
        except ValueError:
            path_text = rest
    path = Path(path_text)
    if not label:
        label = path.stem
    if not np.isfinite(weight) or weight <= 0.0:
        raise ValueError(f"submission weight must be positive and finite: {value}")
    return SubmissionInput(label=_safe_label(label), path=path, weight=float(weight))


def _read_track5_submission_csv(source: Any) -> pd.DataFrame:
    """Read official submission CSV data without coercing opaque identifiers."""

    try:
        return pd.read_csv(source, dtype=str, keep_default_na=False)
    except TypeError:
        return pd.read_csv(source)


def load_track5_submission(path: Path) -> pd.DataFrame:
    """Load an official Track 5 CSV or ZIP submission into normalized columns."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if RESULTS_CSV_NAME in names:
                member = RESULTS_CSV_NAME
            elif len(names) == 1:
                member = names[0]
            else:
                raise ValueError(f"cannot choose Track 5 CSV inside {path}: {names}")
            with archive.open(member) as handle:
                rows = _read_track5_submission_csv(handle)
    else:
        rows = _read_track5_submission_csv(path)
    try:
        rows = _normalize_official_submission_frame(rows, source_path=path)
        return _normalize_submission_rows(rows, source_path=path)
    except ValueError:
        if _has_normalized_submission_columns(rows):
            return _normalize_internal_submission_rows(rows, source_path=path)
        raise


def _normalize_official_submission_frame(rows: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    try:
        return normalize_official_track5_results_frame(rows)
    except ValueError as exc:
        message = str(exc)
        if "Timestamp" in message:
            prefix = "invalid Track 5 Timestamp"
        elif "Classification" in message:
            prefix = "invalid Track 5 Classification"
        elif "Position" in message:
            prefix = "invalid Track 5 Position"
        elif "Sequence" in message:
            prefix = "invalid Track 5 Sequence"
        else:
            prefix = "invalid Track 5 submission"
        raise ValueError(f"{prefix} in {source_path}: {message}") from exc


def ensemble_track5_submissions(
    submissions: Iterable[SubmissionInput],
    *,
    class_policy: str = "weighted-vote",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ensembled estimates and per-row diagnostics."""

    inputs = tuple(submissions)
    if not inputs:
        raise ValueError("at least one submission is required")
    frames: list[pd.DataFrame] = []
    for item in inputs:
        rows = load_track5_submission(item.path)
        rows["ensemble_input_label"] = item.label
        rows["ensemble_input_path"] = str(item.path)
        rows["ensemble_weight"] = float(item.weight)
        frames.append(rows)
    stacked = pd.concat(frames, ignore_index=True, sort=False)
    expected_count = len(frames[0])
    reference_keys = _submission_keys(frames[0])
    for item, rows in zip(inputs, frames, strict=True):
        if len(rows) != expected_count:
            raise ValueError(
                f"submission {item.label} has {len(rows)} rows; expected {expected_count}"
            )
        if _submission_keys(rows) != reference_keys:
            raise ValueError(f"submission {item.label} does not match the reference template keys")

    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in stacked.groupby(["sequence_id", "time_s"], sort=True):
        weights = group["ensemble_weight"].to_numpy(float)
        xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        weighted_xyz = np.sum(weights[:, None] * xyz, axis=0) / float(np.sum(weights))
        classification = _ensemble_classification(group, policy=class_policy)
        spread = _weighted_spread_m(xyz, weights, weighted_xyz)
        records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "source": "track5-submission-ensemble",
                "track_id": "track5-submission-ensemble",
                "state_x_m": float(weighted_xyz[0]),
                "state_y_m": float(weighted_xyz[1]),
                "state_z_m": float(weighted_xyz[2]),
                "Classification": int(classification),
                "ensemble_input_count": int(len(group)),
                "ensemble_weight_sum": float(np.sum(weights)),
                "ensemble_position_spread_m": float(spread),
            }
        )
        diagnostics.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_count": int(len(group)),
                "weight_sum": float(np.sum(weights)),
                "position_spread_m": float(spread),
                "classification": int(classification),
                "classification_vote_margin": float(_classification_vote_margin(group)),
                "input_labels": ";".join(group["ensemble_input_label"].astype(str)),
            }
        )
    estimates = pd.DataFrame.from_records(records).sort_values(["sequence_id", "time_s"])
    diagnostics_df = pd.DataFrame.from_records(diagnostics).sort_values(["sequence_id", "time_s"])
    return estimates.reset_index(drop=True), diagnostics_df.reset_index(drop=True)


def write_track5_submission_ensemble_outputs(
    *,
    estimates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ensembled estimates, official CSV/ZIP, diagnostics, and validation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / ENSEMBLED_ESTIMATES_CSV,
        "results_csv": output / ENSEMBLED_RESULTS_CSV,
        "zip": output / ENSEMBLED_ZIP,
        "diagnostics_csv": output / ENSEMBLE_DIAGNOSTICS_CSV,
        "manifest_json": output / ENSEMBLE_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = estimates.copy()
    official_rows["classification"] = official_rows["Classification"]
    write_official_mmaud_results_csv(
        official_rows,
        paths["results_csv"],
        classification=0,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        official_rows,
        paths["zip"],
        classification=0,
        invalid_row_policy="raise",
    )
    validation_summary: dict[str, Any] | None = None
    if template is not None:
        validation = validate_official_track5_submission(
            paths["zip"],
            template=template,
            require_zip=True,
        )
        paths["validation_json"] = output / VALIDATION_JSON
        paths["validation_rows_csv"] = output / VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-submission-ensemble-v1",
            "row_count": int(len(estimates)),
            "sequence_count": int(estimates["sequence_id"].nunique()) if not estimates.empty else 0,
            "mean_position_spread_m": float(diagnostics["position_spread_m"].mean())
            if not diagnostics.empty
            else None,
            "p95_position_spread_m": float(np.percentile(diagnostics["position_spread_m"], 95))
            if not diagnostics.empty
            else None,
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-ensemble-track5-submissions",
        description="weighted ensemble of official MMUAD/UG2+ Track 5 submissions",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT:PATH",
        help="official CSV/ZIP submission; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="optional official template for preflight validation",
    )
    parser.add_argument(
        "--class-policy",
        choices=("weighted-vote", "first"),
        default="weighted-vote",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        parser.error(
            "--require-leaderboard-ready requires --template so leaderboard readiness can be validated"
        )
    if not args.submission:
        parser.error("provide at least one --submission")
    inputs = tuple(parse_submission_input(value) for value in args.submission)
    estimates, diagnostics = ensemble_track5_submissions(inputs, class_policy=args.class_policy)
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_submission_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        template=template,
        manifest={
            "inputs": [
                {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
                for item in inputs
            ],
            "class_policy": str(args.class_policy),
            "template": None if args.template is None else str(args.template),
        },
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    print("mmuad_track5_submission_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if (
        args.require_leaderboard_ready
        and validation
        and not validation.get("leaderboard_ready", False)
    ):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"ensemble is not leaderboard-ready: {reasons}")
    return 0


def _normalize_submission_rows(rows: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    required = {SEQUENCE_COLUMN, TIME_COLUMN, POSITION_COLUMN, CLASSIFICATION_COLUMN}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{source_path} missing official columns: {missing}")
    out_records: list[dict[str, Any]] = []
    try:
        normalized_classifications = _predicted_class_labels(rows[CLASSIFICATION_COLUMN])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Track 5 Classification values") from exc
    for row, classification_label in zip(
        rows.itertuples(index=False), normalized_classifications, strict=True
    ):
        sequence = str(getattr(row, SEQUENCE_COLUMN))
        timestamp = _parse_timestamp_cell(getattr(row, TIME_COLUMN))
        xyz = _parse_position_cell(getattr(row, POSITION_COLUMN))
        try:
            classification = int(classification_label)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Track 5 Classification values") from exc
        out_records.append(
            {
                "sequence_id": sequence,
                "time_s": timestamp,
                "state_x_m": float(xyz[0]),
                "state_y_m": float(xyz[1]),
                "state_z_m": float(xyz[2]),
                "Classification": classification,
            }
        )
    return (
        pd.DataFrame.from_records(out_records)
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _has_normalized_submission_columns(rows: pd.DataFrame) -> bool:
    lookup = _normalized_column_lookup(rows)
    return all(column in lookup for column in _NORMALIZED_REQUIRED_COLUMNS) and (
        _normalized_classification_column(lookup) is not None
    )


def _normalize_internal_submission_rows(rows: pd.DataFrame, *, source_path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    lookup = _normalized_column_lookup(frame)
    classification_column = _normalized_classification_column(lookup)
    if classification_column is None:
        raise ValueError(f"{source_path} missing normalized Classification/classification column")
    out = pd.DataFrame(
        {
            "sequence_id": frame[lookup["sequence_id"]].astype(str),
            "time_s": pd.to_numeric(frame[lookup["time_s"]], errors="coerce"),
            "state_x_m": pd.to_numeric(frame[lookup["state_x_m"]], errors="coerce"),
            "state_y_m": pd.to_numeric(frame[lookup["state_y_m"]], errors="coerce"),
            "state_z_m": pd.to_numeric(frame[lookup["state_z_m"]], errors="coerce"),
            "Classification": pd.to_numeric(frame[classification_column], errors="coerce"),
        }
    )
    finite = np.isfinite(
        out[["time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"]].to_numpy(float)
    ).all(axis=1)
    out = out.loc[finite].copy()
    if out.empty:
        raise ValueError(f"{source_path} contains no finite normalized submission rows")
    out["Classification"] = out["Classification"].astype(int)
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalized_column_lookup(rows: pd.DataFrame) -> dict[str, Any]:
    return {str(column).strip().lower(): column for column in rows.columns}


def _normalized_classification_column(lookup: dict[str, Any]) -> Any | None:
    for alias in _NORMALIZED_CLASSIFICATION_ALIASES:
        column = lookup.get(alias.strip().lower())
        if column is not None:
            return column
    return None


def _parse_timestamp_cell(value: Any) -> float:
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Track 5 Timestamp cell: {value!r}") from exc
    if not np.isfinite(timestamp):
        raise ValueError(f"invalid Track 5 Timestamp cell: {value!r}")
    return timestamp


def _parse_position_cell(value: Any) -> np.ndarray:
    try:
        return np.asarray(parse_official_position_cell(value), dtype=float)
    except ValueError:
        pass
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"invalid Track 5 Position cell: {value!r}") from exc
    else:
        parsed = value
    if isinstance(parsed, dict):
        for keys in (("x", "y", "z"), ("X", "Y", "Z"), ("state_x_m", "state_y_m", "state_z_m")):
            if all(key in parsed for key in keys):
                return np.asarray([parsed[key] for key in keys], dtype=float)
        raise ValueError(f"unsupported Position object keys: {sorted(parsed)}")
    try:
        array = np.asarray(parsed, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Track 5 Position cell: {value!r}") from exc
    if len(array) != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid Track 5 Position cell: {value!r}")
    return array.astype(float)


def _submission_keys(rows: pd.DataFrame) -> list[tuple[str, float]]:
    return list(zip(rows["sequence_id"].astype(str), rows["time_s"].astype(float), strict=True))


def _ensemble_classification(group: pd.DataFrame, *, policy: str) -> int:
    if policy == "first":
        return int(group.iloc[0]["Classification"])
    votes: dict[int, float] = {}
    for row in group.itertuples(index=False):
        label = int(getattr(row, "Classification"))
        votes[label] = votes.get(label, 0.0) + float(getattr(row, "ensemble_weight"))
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _classification_vote_margin(group: pd.DataFrame) -> float:
    votes: dict[int, float] = {}
    for row in group.itertuples(index=False):
        label = int(getattr(row, "Classification"))
        votes[label] = votes.get(label, 0.0) + float(getattr(row, "ensemble_weight"))
    ordered = sorted(votes.values(), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    return float(ordered[0] - ordered[1])


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    deltas = np.linalg.norm(np.asarray(xyz, dtype=float) - np.asarray(center, dtype=float), axis=1)
    return float(np.sum(np.asarray(weights, dtype=float) * deltas) / float(np.sum(weights)))


def _safe_label(value: object) -> str:
    return str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_") or "submission"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
