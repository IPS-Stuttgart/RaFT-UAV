"""Resample MMUAD estimates to an official Track 5 timestamp template.

Candidate-mixture and tracker outputs often live on sensor/candidate timestamps,
whereas Codabench submissions must contain exactly the requested
``Sequence,Timestamp`` rows.  This helper interpolates one trajectory per
sequence onto an official template, writes upload-ready Track 5 artifacts, and
runs the same local preflight validator used by the scorecard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    parse_official_classification_cell,
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

COORDINATE_COLUMN_SETS = (
    ("state_x_m", "state_y_m", "state_z_m"),
    ("x_m", "y_m", "z_m"),
    ("x", "y", "z"),
    ("east_m", "north_m", "up_m"),
)
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "timestamp_s", "time")
CLASSIFICATION_ALIASES = (
    "classification",
    "Classification",
    "class_id",
    "class",
    "label",
    "uav_type",
    "uav_type_id",
)
RESAMPLE_METHODS = ("linear", "nearest")
CLASSIFICATION_POLICIES = ("sequence-mode", "nearest", "none")
ResampleMethod = Literal["linear", "nearest"]
ClassificationPolicy = Literal["sequence-mode", "nearest", "none"]


def resample_estimates_to_track5_template(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate estimate coordinates onto official template timestamps.

    Returns ``(resampled_estimates, diagnostics)``.  ``linear`` interpolation
    uses endpoint hold outside each sequence's estimate time span, matching the
    pragmatic leaderboard-submission need to produce one prediction for every
    requested timestamp while making extrapolated rows auditable through
    diagnostics.  ``nearest`` is useful when tracker outputs are already aligned
    to a template-like grid or when long-gap interpolation is unsafe.

    When the estimate table contains numeric Track 5 class ids, the resampled
    rows also carry a ``classification`` column.  By default the sequence-level
    mode is used, which preserves fused sequence classifiers through final
    template packaging without requiring a separate class-map file.
    """

    method = _normalize_resample_method(resample_method)
    class_policy = _normalize_classification_policy(classification_policy)
    estimate_rows = _normalize_estimate_rows(estimates)
    template_rows = _normalize_template_rows(template)
    if template_rows.empty:
        return _empty_resampled(), _empty_diagnostics()

    estimate_by_sequence = {
        sequence_id: group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in estimate_rows.groupby("sequence_id", sort=True)
    }
    resampled_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        group = estimate_by_sequence.get(sequence_id)
        if group is None or group.empty:
            record = _missing_estimate_record(sequence_id, time_s, method, class_policy)
            resampled_records.append(record)
            diagnostic_records.append(
                _diagnostic_record(
                    sequence_id=sequence_id,
                    time_s=time_s,
                    nearest_time_delta_s=np.nan,
                    extrapolated=True,
                    source_row_count=0,
                    valid=False,
                    resample_method=method,
                    interpolation_gap_s=np.nan,
                    large_gap_fallback=False,
                    classification_source="missing_sequence",
                    classification_policy=class_policy,
                )
            )
            continue
        interp = _resampled_position(
            group,
            time_s,
            resample_method=method,
            max_interpolation_gap_s=max_interpolation_gap_s,
        )
        classification, classification_source = _resampled_classification(
            group,
            time_s,
            classification_policy=class_policy,
        )
        valid = np.isfinite(interp.position).all()
        if max_nearest_time_delta_s is not None and np.isfinite(interp.nearest_delta_s):
            valid = valid and abs(float(interp.nearest_delta_s)) <= float(max_nearest_time_delta_s)
        record = {
            "sequence_id": sequence_id,
            "time_s": time_s,
            "state_x_m": float(interp.position[0])
            if np.isfinite(interp.position[0])
            else np.nan,
            "state_y_m": float(interp.position[1])
            if np.isfinite(interp.position[1])
            else np.nan,
            "state_z_m": float(interp.position[2])
            if np.isfinite(interp.position[2])
            else np.nan,
            "template_resampled": True,
            "template_nearest_time_delta_s": float(interp.nearest_delta_s),
            "template_extrapolated": bool(interp.extrapolated),
            "template_resample_valid": bool(valid),
            "template_resample_method": interp.method_used,
            "template_interpolation_gap_s": interp.interpolation_gap_s,
            "template_large_gap_fallback": bool(interp.large_gap_fallback),
            "template_classification_policy": class_policy,
            "template_classification_source": classification_source,
        }
        if classification is not None:
            record["classification"] = int(classification)
        resampled_records.append(record)
        diagnostic_records.append(
            _diagnostic_record(
                sequence_id=sequence_id,
                time_s=time_s,
                nearest_time_delta_s=interp.nearest_delta_s,
                extrapolated=interp.extrapolated,
                source_row_count=len(group),
                valid=valid,
                resample_method=interp.method_used,
                interpolation_gap_s=interp.interpolation_gap_s,
                large_gap_fallback=interp.large_gap_fallback,
                classification_source=classification_source,
                classification_policy=class_policy,
            )
        )
    resampled = pd.DataFrame.from_records(resampled_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return resampled, diagnostics


def summarize_template_resample_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return per-sequence coverage diagnostics for template resampling."""

    rows = pd.DataFrame(diagnostics).copy()
    columns = [
        "sequence_id",
        "template_row_count",
        "valid_row_count",
        "invalid_row_count",
        "valid_fraction",
        "extrapolated_row_count",
        "extrapolated_fraction",
        "large_gap_fallback_row_count",
        "large_gap_fallback_fraction",
        "linear_method_row_count",
        "nearest_method_row_count",
        "source_row_count_min",
        "source_row_count_max",
        "nearest_time_delta_abs_mean_s",
        "nearest_time_delta_abs_p95_s",
        "nearest_time_delta_abs_max_s",
        "interpolation_gap_mean_s",
        "interpolation_gap_p95_s",
        "interpolation_gap_max_s",
        "classification_mode_row_count",
        "classification_nearest_row_count",
        "classification_missing_row_count",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    if "sequence_id" not in rows.columns:
        raise ValueError("diagnostics must include sequence_id")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["valid"] = _bool_column(rows, "valid")
    rows["extrapolated"] = _bool_column(rows, "extrapolated")
    rows["large_gap_fallback"] = _bool_column(rows, "large_gap_fallback")
    rows["resample_method"] = (
        _series_or_default(rows, "resample_method", "linear").fillna("linear").astype(str)
    )
    rows["classification_source"] = (
        _series_or_default(rows, "classification_source", "none").fillna("none").astype(str)
    )
    rows["source_row_count"] = pd.to_numeric(
        _series_or_default(rows, "source_row_count", 0),
        errors="coerce",
    ).fillna(0)
    rows["abs_nearest_time_delta_s"] = _nearest_delta_abs_series(rows)
    rows["interpolation_gap_s"] = pd.to_numeric(
        _series_or_default(rows, "interpolation_gap_s", np.nan),
        errors="coerce",
    )
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        row_count = int(len(group))
        valid_count = int(group["valid"].sum())
        extrapolated_count = int(group["extrapolated"].sum())
        large_gap_count = int(group["large_gap_fallback"].sum())
        abs_delta = group["abs_nearest_time_delta_s"].dropna()
        interpolation_gap = group["interpolation_gap_s"].dropna()
        classification_source = group["classification_source"]
        records.append(
            {
                "sequence_id": str(sequence_id),
                "template_row_count": row_count,
                "valid_row_count": valid_count,
                "invalid_row_count": int(row_count - valid_count),
                "valid_fraction": _fraction(valid_count, row_count),
                "extrapolated_row_count": extrapolated_count,
                "extrapolated_fraction": _fraction(extrapolated_count, row_count),
                "large_gap_fallback_row_count": large_gap_count,
                "large_gap_fallback_fraction": _fraction(large_gap_count, row_count),
                "linear_method_row_count": int((group["resample_method"] == "linear").sum()),
                "nearest_method_row_count": int((group["resample_method"] == "nearest").sum()),
                "source_row_count_min": int(group["source_row_count"].min()),
                "source_row_count_max": int(group["source_row_count"].max()),
                "nearest_time_delta_abs_mean_s": _mean(abs_delta),
                "nearest_time_delta_abs_p95_s": _quantile(abs_delta, 0.95),
                "nearest_time_delta_abs_max_s": _max(abs_delta),
                "interpolation_gap_mean_s": _mean(interpolation_gap),
                "interpolation_gap_p95_s": _quantile(interpolation_gap, 0.95),
                "interpolation_gap_max_s": _max(interpolation_gap),
                "classification_mode_row_count": int((classification_source == "sequence-mode").sum()),
                "classification_nearest_row_count": int((classification_source == "nearest").sum()),
                "classification_missing_row_count": int(
                    classification_source.isin({"none", "missing", "missing_sequence"}).sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def write_track5_template_resample_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
) -> dict[str, Path]:
    """Write resampled estimates, official CSV/ZIP, diagnostics, and validation."""

    method = _normalize_resample_method(resample_method)
    class_policy = _normalize_classification_policy(classification_policy)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resampled, diagnostics = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=class_policy,
    )
    resampled_for_export = _fill_missing_resampled_classification(
        resampled,
        default_classification=default_classification,
    )
    diagnostics_by_sequence = summarize_template_resample_diagnostics(diagnostics)
    paths = {
        "resampled_estimates_csv": output / "mmuad_template_resampled_estimates.csv",
        "diagnostics_csv": output / "mmuad_template_resample_diagnostics.csv",
        "diagnostics_by_sequence_csv": (
            output / "mmuad_template_resample_diagnostics_by_sequence.csv"
        ),
        "official_results_csv": output / "mmaud_results.csv",
        "official_zip": output / "ug2_submission.zip",
        "validation_json": output / "mmuad_template_resample_validation.json",
        "validation_rows_csv": output / "mmuad_template_resample_validation_rows.csv",
        "manifest_json": output / "mmuad_template_resample_manifest.json",
    }
    resampled_for_export.to_csv(paths["resampled_estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    diagnostics_by_sequence.to_csv(paths["diagnostics_by_sequence_csv"], index=False)
    write_official_mmaud_results_csv(
        resampled_for_export,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        resampled_for_export,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(
        paths["official_zip"],
        template=template,
        require_zip=True,
    )
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    valid_rows = resampled.get("template_resample_valid", pd.Series(dtype=bool))
    extrapolated_rows = resampled.get("template_extrapolated", pd.Series(dtype=bool))
    fallback_rows = resampled.get("template_large_gap_fallback", pd.Series(dtype=bool))
    classification_rows = pd.to_numeric(
        _series_or_default(resampled_for_export, "classification", np.nan),
        errors="coerce",
    ).notna()
    manifest = {
        "schema": "raft-uav-mmuad-track5-template-resample-v3",
        "row_count": int(len(resampled)),
        "template_row_count": int(len(_normalize_template_rows(template))),
        "sequence_count": int(len(diagnostics_by_sequence)),
        "valid_resampled_rows": int(valid_rows.sum()),
        "invalid_resampled_rows": int((~valid_rows.astype(bool)).sum()),
        "extrapolated_rows": int(extrapolated_rows.sum()),
        "large_gap_fallback_rows": int(fallback_rows.sum()),
        "resampled_classification_rows": int(classification_rows.sum()),
        "invalid_sequence_count": int(
            (diagnostics_by_sequence.get("invalid_row_count", pd.Series(dtype=int)) > 0).sum()
        ),
        "extrapolated_sequence_count": int(
            (
                diagnostics_by_sequence.get(
                    "extrapolated_row_count",
                    pd.Series(dtype=int),
                )
                > 0
            ).sum()
        ),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "default_classification": str(default_classification),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "resample_method": method,
        "max_interpolation_gap_s": max_interpolation_gap_s,
        "classification_policy": class_policy,
        "classification_from_estimates": bool("classification" in resampled.columns),
        "class_map_applied": bool(class_map),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


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

    estimates = pd.read_csv(args.estimates_csv)
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


def _normalize_estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
        )
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    coord_columns = _coordinate_columns(rows)
    classification_column = _first_present(rows, CLASSIFICATION_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("estimates must contain sequence and time columns")
    out = pd.DataFrame(
        {
            "sequence_id": _normalized_sequence_values(rows[sequence_column]),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "state_x_m": pd.to_numeric(rows[coord_columns[0]], errors="coerce"),
            "state_y_m": pd.to_numeric(rows[coord_columns[1]], errors="coerce"),
            "state_z_m": pd.to_numeric(rows[coord_columns[2]], errors="coerce"),
        }
    )
    if classification_column is not None:
        out["classification"] = _normalized_classification_values(rows[classification_column])
    finite = out["sequence_id"].notna()
    finite &= np.isfinite(
        out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": _normalized_sequence_values(rows[sequence_column]),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _coordinate_columns(rows: pd.DataFrame) -> tuple[str, str, str]:
    for columns in COORDINATE_COLUMN_SETS:
        if all(column in rows.columns for column in columns):
            return columns
    expected = " or ".join(",".join(columns) for columns in COORDINATE_COLUMN_SETS)
    raise ValueError(f"estimates must contain coordinate columns: {expected}")


class _TemplatePosition:
    def __init__(
        self,
        *,
        position: np.ndarray,
        nearest_delta_s: float,
        extrapolated: bool,
        method_used: ResampleMethod,
        interpolation_gap_s: float,
        large_gap_fallback: bool,
    ) -> None:
        self.position = position
        self.nearest_delta_s = nearest_delta_s
        self.extrapolated = extrapolated
        self.method_used = method_used
        self.interpolation_gap_s = interpolation_gap_s
        self.large_gap_fallback = large_gap_fallback


def _resampled_position(
    group: pd.DataFrame,
    time_s: float,
    *,
    resample_method: ResampleMethod,
    max_interpolation_gap_s: float | None,
) -> _TemplatePosition:
    work = group.sort_values("time_s").drop_duplicates("time_s", keep="last")
    times = work["time_s"].to_numpy(float)
    xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    nearest_idx = int(np.argmin(np.abs(times - float(time_s))))
    nearest_delta = float(float(time_s) - times[nearest_idx])
    extrapolated = bool(float(time_s) < times[0] or float(time_s) > times[-1])
    interpolation_gap_s = _bracketing_gap_s(times, float(time_s))
    if len(times) == 1 or resample_method == "nearest":
        return _TemplatePosition(
            position=xyz[nearest_idx].astype(float),
            nearest_delta_s=nearest_delta,
            extrapolated=extrapolated,
            method_used="nearest",
            interpolation_gap_s=interpolation_gap_s,
            large_gap_fallback=False,
        )
    large_gap_fallback = (
        max_interpolation_gap_s is not None
        and np.isfinite(interpolation_gap_s)
        and float(interpolation_gap_s) > float(max_interpolation_gap_s)
    )
    if large_gap_fallback:
        return _TemplatePosition(
            position=xyz[nearest_idx].astype(float),
            nearest_delta_s=nearest_delta,
            extrapolated=extrapolated,
            method_used="nearest",
            interpolation_gap_s=interpolation_gap_s,
            large_gap_fallback=True,
        )
    interpolated = np.asarray(
        [np.interp(float(time_s), times, xyz[:, axis]) for axis in range(3)]
    )
    return _TemplatePosition(
        position=interpolated,
        nearest_delta_s=nearest_delta,
        extrapolated=extrapolated,
        method_used="linear",
        interpolation_gap_s=interpolation_gap_s,
        large_gap_fallback=False,
    )


def _interpolated_position(group: pd.DataFrame, time_s: float) -> tuple[np.ndarray, float, bool]:
    """Backward-compatible wrapper for older tests/imports."""

    result = _resampled_position(
        group,
        time_s,
        resample_method="linear",
        max_interpolation_gap_s=None,
    )
    return result.position, result.nearest_delta_s, result.extrapolated


def _resampled_classification(
    group: pd.DataFrame,
    time_s: float,
    *,
    classification_policy: ClassificationPolicy,
) -> tuple[int | None, str]:
    if classification_policy == "none" or "classification" not in group.columns:
        return None, "none"
    valid = group.loc[pd.to_numeric(group["classification"], errors="coerce").notna()].copy()
    if valid.empty:
        return None, "missing"
    valid["classification"] = pd.to_numeric(valid["classification"], errors="coerce").astype(int)
    if classification_policy == "sequence-mode":
        mode = valid["classification"].mode(dropna=True)
        if not mode.empty:
            return int(mode.sort_values().iloc[0]), "sequence-mode"
    nearest_idx = (pd.to_numeric(valid["time_s"], errors="coerce") - float(time_s)).abs().idxmin()
    return int(valid.loc[nearest_idx, "classification"]), "nearest"


def _bracketing_gap_s(times: np.ndarray, time_s: float) -> float:
    if len(times) < 2:
        return np.nan
    if float(time_s) <= float(times[0]) or float(time_s) >= float(times[-1]):
        return np.nan
    right = int(np.searchsorted(times, float(time_s), side="right"))
    left = max(0, right - 1)
    if right >= len(times):
        return np.nan
    return float(times[right] - times[left])


def _missing_estimate_record(
    sequence_id: str,
    time_s: float,
    resample_method: str,
    classification_policy: str,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "state_x_m": np.nan,
        "state_y_m": np.nan,
        "state_z_m": np.nan,
        "template_resampled": True,
        "template_nearest_time_delta_s": np.nan,
        "template_extrapolated": True,
        "template_resample_valid": False,
        "template_resample_method": resample_method,
        "template_interpolation_gap_s": np.nan,
        "template_large_gap_fallback": False,
        "template_classification_policy": classification_policy,
        "template_classification_source": "missing_sequence",
    }


def _diagnostic_record(
    *,
    sequence_id: str,
    time_s: float,
    nearest_time_delta_s: float,
    extrapolated: bool,
    source_row_count: int,
    valid: bool,
    resample_method: str,
    interpolation_gap_s: float,
    large_gap_fallback: bool,
    classification_source: str,
    classification_policy: str,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "nearest_time_delta_s": nearest_time_delta_s,
        "abs_nearest_time_delta_s": (
            abs(float(nearest_time_delta_s))
            if np.isfinite(nearest_time_delta_s)
            else np.nan
        ),
        "extrapolated": bool(extrapolated),
        "source_row_count": int(source_row_count),
        "valid": bool(valid),
        "resample_method": str(resample_method),
        "interpolation_gap_s": interpolation_gap_s,
        "large_gap_fallback": bool(large_gap_fallback),
        "classification_source": str(classification_source),
        "classification_policy": str(classification_policy),
    }


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    value = rows[column]
    if value.dtype == bool:
        return value.fillna(False).astype(bool)
    return value.fillna(False).map(lambda item: str(item).lower() in {"1", "true", "yes"})


def _series_or_default(rows: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in rows.columns:
        return rows[column]
    return pd.Series(default, index=rows.index)


def _nearest_delta_abs_series(rows: pd.DataFrame) -> pd.Series:
    if "abs_nearest_time_delta_s" in rows.columns:
        return pd.to_numeric(rows["abs_nearest_time_delta_s"], errors="coerce")
    nearest_delta = pd.to_numeric(
        _series_or_default(rows, "nearest_time_delta_s", np.nan),
        errors="coerce",
    )
    return nearest_delta.abs()


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else float("nan")


def _max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else float("nan")


def _empty_resampled() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "state_x_m",
            "state_y_m",
            "state_z_m",
        ]
    )


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "nearest_time_delta_s",
            "extrapolated",
            "source_row_count",
            "valid",
            "resample_method",
            "interpolation_gap_s",
            "large_gap_fallback",
            "classification_source",
            "classification_policy",
        ]
    )


def _normalized_sequence_values(values: pd.Series) -> pd.Series:
    return values.map(_sequence_text_or_none)


def _sequence_text_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _normalized_classification_values(values: pd.Series) -> pd.Series:
    return values.map(_classification_or_none)


def _classification_or_none(value: Any) -> int | None:
    try:
        return parse_official_classification_cell(value)
    except ValueError:
        return None


def _fill_missing_resampled_classification(
    resampled: pd.DataFrame,
    *,
    default_classification: int | str,
) -> pd.DataFrame:
    rows = pd.DataFrame(resampled).copy()
    if "classification" not in rows.columns:
        return rows
    default_value = parse_official_classification_cell(default_classification)
    values = pd.to_numeric(rows["classification"], errors="coerce")
    rows["classification"] = values.where(values.notna(), default_value).astype(int)
    return rows


def _first_present(rows: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for candidate in candidates:
        if candidate in rows.columns:
            return candidate
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


def _normalize_resample_method(method: str) -> ResampleMethod:
    value = str(method).strip().lower()
    if value not in RESAMPLE_METHODS:
        raise ValueError(f"resample_method must be one of {RESAMPLE_METHODS}; got {method!r}")
    return value  # type: ignore[return-value]


def _normalize_classification_policy(policy: str) -> ClassificationPolicy:
    value = str(policy).strip().lower()
    if value not in CLASSIFICATION_POLICIES:
        raise ValueError(
            f"classification_policy must be one of {CLASSIFICATION_POLICIES}; got {policy!r}"
        )
    return value  # type: ignore[return-value]


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
