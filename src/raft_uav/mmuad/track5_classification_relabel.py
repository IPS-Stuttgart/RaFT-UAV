"""Copy Track 5 classification labels onto another official submission."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    OFFICIAL_TRACK5_CLASS_IDS,
    load_official_track5_results_frame,
    load_official_track5_template_file,
    validate_official_track5_submission,
)


RELABEL_RESULTS_CSV = "mmaud_results_relabelled.csv"
RELABEL_ZIP = "ug2_submission_relabelled.zip"
RELABEL_DIAGNOSTICS_CSV = "mmuad_track5_classification_relabel_diagnostics.csv"
RELABEL_MANIFEST_JSON = "mmuad_track5_classification_relabel_manifest.json"
RELABEL_VALIDATION_JSON = "mmuad_track5_classification_relabel_validation.json"
RelabelMode = Literal["by-key", "by-sequence-majority", "by-nearest-time"]
VALID_CLASS_IDS = tuple(sorted(OFFICIAL_TRACK5_CLASS_IDS))
MIN_PROBABILITY_CLASS_COUNT = 2
SEQUENCE_ALIASES = ("Sequence", "sequence_id", "sequence", "heldout_sequence", "seq")
PREDICTED_CLASS_ALIASES = (
    "predicted_class",
    "Classification",
    "classification",
    "uav_type",
    "class_id",
    "label",
)
PROBABILITY_PREFIXES = (
    "predicted_probability_",
    "class_prob_",
    "class_probability_",
    "probability_",
    "p_class_",
    "prob_class_",
)


@dataclass(frozen=True)
class ClassificationRelabelResult:
    """Official relabelled rows plus diagnostics."""

    rows: pd.DataFrame
    diagnostics: pd.DataFrame
    manifest: dict[str, Any]


def relabel_track5_classification(
    pose_submission: pd.DataFrame,
    classification_submission: pd.DataFrame,
    *,
    mode: RelabelMode = "by-key",
    max_nearest_time_delta_s: float | None = None,
) -> ClassificationRelabelResult:
    """Return pose rows with labels copied from another official submission.

    ``by-key`` requires exactly matching ``Sequence``/``Timestamp`` rows.
    ``by-sequence-majority`` copies one majority label per sequence.  The
    ``by-nearest-time`` mode is useful when the classifier submission was
    generated on a dense sensor-time grid rather than the final Codabench
    template; it copies the closest same-sequence label and records the time
    offset in the diagnostics.
    """

    pose = _normalize_frame(pose_submission, name="pose_submission")
    source = _normalize_frame(classification_submission, name="classification_submission")
    if mode == "by-key":
        labels = source[["Sequence", "Timestamp", "Classification"]].rename(
            columns={"Classification": "source_classification"},
        )
        merged = pose.merge(
            labels,
            on=["Sequence", "Timestamp"],
            how="left",
            validate="one_to_one",
        )
    elif mode == "by-sequence-majority":
        labels = (
            source.groupby("Sequence", sort=True)["Classification"]
            .agg(_majority_class)
            .rename("source_classification")
            .reset_index()
        )
        merged = pose.merge(labels, on="Sequence", how="left", validate="many_to_one")
    elif mode == "by-nearest-time":
        merged = _nearest_time_relabel_merge(
            pose,
            source,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
    else:
        raise ValueError(
            "classification relabel mode must be 'by-key', 'by-sequence-majority', "
            "or 'by-nearest-time'",
        )
    result = _build_relabel_result(
        pose,
        merged,
        mode=str(mode),
        source_kind="official-submission",
    )
    if mode == "by-nearest-time":
        result.manifest["max_nearest_time_delta_s"] = max_nearest_time_delta_s
    return result


def relabel_track5_classification_from_sequence_predictions(
    pose_submission: pd.DataFrame,
    sequence_predictions: pd.DataFrame,
) -> ClassificationRelabelResult:
    """Return pose rows with labels copied from sequence-level prediction rows.

    The prediction table may contain one predicted class per sequence, official
    ``Classification`` labels, or probability columns such as
    ``predicted_probability_0`` ... ``predicted_probability_3``.  Probability
    files may include all classes or any subset of at least two official class
    IDs; missing classes are treated as unavailable candidates.  The legacy
    four-column ``0`` ... ``3`` probability layout is still accepted.
    Probability rows are averaged per sequence before taking argmax.  This makes
    it easy to combine a strong pose submission with a non-image/image-fused
    sequence classifier without first fabricating an official-style submission.
    """

    pose = _normalize_frame(pose_submission, name="pose_submission")
    labels = _sequence_prediction_labels(sequence_predictions)
    merged = pose.merge(labels, on="Sequence", how="left", validate="many_to_one")
    return _build_relabel_result(
        pose,
        merged,
        mode="by-sequence-prediction",
        source_kind="sequence-predictions",
    )


def write_track5_classification_relabel_outputs(
    *,
    result: ClassificationRelabelResult,
    output_dir: Path,
    pose_submission_path: Path,
    classification_submission_path: Path,
    template: pd.DataFrame | None = None,
    require_leaderboard_ready: bool = False,
    classification_source_kind: str = "official-submission",
) -> dict[str, Path]:
    """Write relabelled official CSV/ZIP plus manifest and validation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_csv": output / RELABEL_RESULTS_CSV,
        "zip": output / RELABEL_ZIP,
        "diagnostics_csv": output / RELABEL_DIAGNOSTICS_CSV,
        "manifest_json": output / RELABEL_MANIFEST_JSON,
    }
    result.rows.to_csv(paths["results_csv"], index=False)
    with ZipFile(paths["zip"], "w", compression=ZIP_DEFLATED) as archive:
        archive.write(paths["results_csv"], arcname="mmaud_results.csv")
    result.diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    validation_summary: dict[str, Any] | None = None
    if template is not None:
        validation = validate_official_track5_submission(
            paths["zip"],
            template=template,
            require_zip=True,
        )
        validation_summary = _jsonable(validation.summary)
        paths["validation_json"] = output / RELABEL_VALIDATION_JSON
        paths["validation_json"].write_text(
            json.dumps(validation_summary, indent=2),
            encoding="utf-8",
        )
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"relabelled submission is not leaderboard-ready: {reasons}")
    manifest = dict(result.manifest)
    manifest.update(
        {
            "pose_submission": str(pose_submission_path),
            "classification_submission": str(classification_submission_path),
            "classification_source": str(classification_submission_path),
            "classification_source_kind": str(classification_source_kind),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-classification-relabel",
        description=__doc__,
    )
    parser.add_argument("--pose-submission", type=Path, required=True)
    parser.add_argument("--classification-submission", type=Path)
    parser.add_argument(
        "--classification-predictions",
        type=Path,
        help=(
            "sequence-level classifier predictions CSV; accepts sequence_id/Sequence and either "
            "predicted_class/Classification, probability columns for at least two official classes "
            "(for example predicted_probability_0..3), or bare 0..3 probability columns"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("by-key", "by-sequence-majority", "by-nearest-time"),
        default="by-key",
    )
    parser.add_argument(
        "--max-nearest-time-delta-s",
        type=float,
        help="maximum same-sequence time offset allowed in by-nearest-time mode",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if (args.classification_submission is None) == (args.classification_predictions is None):
        parser.error(
            "provide exactly one of --classification-submission or --classification-predictions",
        )
    if args.max_nearest_time_delta_s is not None and args.mode != "by-nearest-time":
        parser.error("--max-nearest-time-delta-s is only valid with --mode by-nearest-time")

    pose_rows = load_official_track5_results_frame(args.pose_submission)
    if args.classification_predictions is not None:
        source_path = args.classification_predictions
        result = relabel_track5_classification_from_sequence_predictions(
            pose_rows,
            _read_sequence_predictions_csv(args.classification_predictions),
        )
        source_kind = "sequence-predictions"
    else:
        source_path = args.classification_submission
        assert source_path is not None
        result = relabel_track5_classification(
            pose_rows,
            load_official_track5_results_frame(source_path),
            mode=args.mode,
            max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        )
        source_kind = "official-submission"
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_classification_relabel_outputs(
        result=result,
        output_dir=args.output_dir,
        pose_submission_path=args.pose_submission,
        classification_submission_path=source_path,
        template=template,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
        classification_source_kind=source_kind,
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_classification_relabel=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    validation = manifest.get("validation") or {}
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    return 0


def _build_relabel_result(
    pose: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    mode: str,
    source_kind: str,
) -> ClassificationRelabelResult:
    if merged["source_classification"].isna().any():
        missing = merged.loc[merged["source_classification"].isna(), ["Sequence", "Timestamp"]]
        raise ValueError(f"classification source is missing {len(missing)} pose rows")
    _validate_class_series(merged["source_classification"], name="source_classification")
    diagnostics = merged[
        ["Sequence", "Timestamp", "Classification", "source_classification"]
    ].copy()
    diagnostics.rename(columns={"Classification": "pose_classification"}, inplace=True)
    diagnostics["relabelled_classification"] = diagnostics["source_classification"].astype(int)
    diagnostics["classification_changed"] = (
        diagnostics["pose_classification"].astype(int)
        != diagnostics["relabelled_classification"].astype(int)
    )
    if "source_time_delta_s" in merged.columns:
        diagnostics["source_time_delta_s"] = merged["source_time_delta_s"]
        diagnostics["source_abs_time_delta_s"] = merged["source_abs_time_delta_s"]
    if "source_classification_probability" in merged.columns:
        diagnostics["source_classification_probability"] = merged[
            "source_classification_probability"
        ]
    if "source_sequence_label_method" in merged.columns:
        diagnostics["source_sequence_label_method"] = merged["source_sequence_label_method"]
    out = pose.copy()
    out["Classification"] = merged["source_classification"].astype(int)
    manifest = {
        "schema": "raft-uav-mmuad-track5-classification-relabel-v1",
        "mode": str(mode),
        "classification_source_kind": str(source_kind),
        "row_count": int(len(out)),
        "sequence_count": int(out["Sequence"].nunique()) if not out.empty else 0,
        "changed_row_count": int(diagnostics["classification_changed"].sum()),
        "changed_sequence_count": int(
            diagnostics.loc[diagnostics["classification_changed"], "Sequence"].nunique()
        ),
    }
    if "source_time_delta_s" in diagnostics.columns:
        abs_delta = pd.to_numeric(diagnostics["source_abs_time_delta_s"], errors="coerce")
        manifest["source_time_delta_abs_mean_s"] = _finite_mean(abs_delta)
        manifest["source_time_delta_abs_max_s"] = _finite_max(abs_delta)
    if "source_classification_probability" in diagnostics.columns:
        probability = pd.to_numeric(
            diagnostics["source_classification_probability"],
            errors="coerce",
        )
        manifest["source_probability_mean"] = _finite_mean(probability)
        manifest["source_probability_min"] = _finite_min(probability)
    return ClassificationRelabelResult(
        rows=out[["Sequence", "Timestamp", "Position", "Classification"]],
        diagnostics=diagnostics,
        manifest=manifest,
    )


def _nearest_time_relabel_merge(
    pose: pd.DataFrame,
    source: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None,
) -> pd.DataFrame:
    if max_nearest_time_delta_s is not None and max_nearest_time_delta_s < 0.0:
        raise ValueError("max_nearest_time_delta_s must be non-negative")
    source_by_sequence = {
        sequence_id: group.sort_values("Timestamp").reset_index(drop=True)
        for sequence_id, group in source.groupby("Sequence", sort=True)
    }
    records: list[dict[str, Any]] = []
    for _, row in pose.iterrows():
        sequence = str(row["Sequence"])
        timestamp = float(row["Timestamp"])
        group = source_by_sequence.get(sequence)
        if group is None or group.empty:
            source_classification = np.nan
            source_time_delta_s = np.nan
        else:
            deltas = pd.to_numeric(group["Timestamp"], errors="coerce") - timestamp
            abs_deltas = deltas.abs()
            nearest_index = int(abs_deltas.idxmin())
            source_time_delta_s = float(deltas.loc[nearest_index])
            if (
                max_nearest_time_delta_s is not None
                and abs(source_time_delta_s) > max_nearest_time_delta_s
            ):
                source_classification = np.nan
            else:
                source_classification = int(group.loc[nearest_index, "Classification"])
        record = row.to_dict()
        record["source_classification"] = source_classification
        record["source_time_delta_s"] = source_time_delta_s
        if np.isfinite(source_time_delta_s):
            record["source_abs_time_delta_s"] = abs(source_time_delta_s)
        else:
            record["source_abs_time_delta_s"] = np.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def _read_sequence_predictions_csv(path: Path | str) -> pd.DataFrame:
    """Read sequence predictions without coercing opaque sequence identifiers."""

    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        return pd.read_csv(path)


def _sequence_prediction_labels(sequence_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(sequence_predictions).copy()
    if rows.empty:
        raise ValueError("sequence prediction table is empty")
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("sequence prediction table missing Sequence/sequence_id column")
    rows["Sequence"] = rows[sequence_column].astype(str).str.strip()
    probability_items = _probability_columns(rows)
    probability_class_ids = tuple(class_id for class_id, _column in probability_items)
    if _valid_probability_class_ids(probability_class_ids):
        probability_columns = [column for _class_id, column in probability_items]
        probability_rows = rows[["Sequence", *probability_columns]].copy()
        for column in probability_columns:
            probability_rows[column] = pd.to_numeric(probability_rows[column], errors="coerce")
        grouped = probability_rows.groupby("Sequence", sort=True)[probability_columns].mean()
        probs = grouped.to_numpy(float)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        totals = probs.sum(axis=1, keepdims=True)
        probs = np.divide(
            probs,
            totals,
            out=np.full_like(probs, 1.0 / len(probability_class_ids), dtype=float),
            where=totals > 0.0,
        )
        class_ids = np.asarray(probability_class_ids, dtype=int)
        predicted = class_ids[np.argmax(probs, axis=1)]
        probability = np.max(probs, axis=1)
        out = pd.DataFrame(
            {
                "Sequence": grouped.index.astype(str),
                "source_classification": predicted.astype(int),
                "source_classification_probability": probability.astype(float),
                "source_sequence_label_method": "probability-argmax",
            }
        )
    else:
        class_column = _first_present(rows, PREDICTED_CLASS_ALIASES)
        if class_column is None:
            raise ValueError(
                "sequence prediction table needs at least two official-class probability "
                "columns or a predicted_class/Classification column"
            )
        labels = rows[["Sequence", class_column]].copy()
        labels[class_column] = pd.to_numeric(labels[class_column], errors="coerce")
        _validate_class_series(labels[class_column], name=class_column)
        out = (
            labels.groupby("Sequence", sort=True)[class_column]
            .agg(_majority_class)
            .rename("source_classification")
            .reset_index()
        )
        out["source_classification_probability"] = np.nan
        out["source_sequence_label_method"] = "class-majority"
    _validate_class_series(out["source_classification"], name="source_classification")
    return out


def _normalize_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    missing = {"Sequence", "Timestamp", "Position", "Classification"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing official columns: {sorted(missing)}")
    out = frame[["Sequence", "Timestamp", "Position", "Classification"]].copy()
    out["Sequence"] = out["Sequence"].astype(str)
    out["Timestamp"] = pd.to_numeric(out["Timestamp"], errors="coerce")
    out["Classification"] = pd.to_numeric(out["Classification"], errors="coerce")
    if not np.isfinite(out[["Timestamp", "Classification"]].to_numpy(float)).all():
        raise ValueError(f"{name} contains non-finite Timestamp or Classification")
    _validate_class_series(out["Classification"], name=f"{name}.Classification")
    return out.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _majority_class(values: pd.Series) -> int:
    counts = values.astype(int).value_counts()
    if counts.empty:
        raise ValueError("cannot compute majority class for empty sequence")
    max_count = counts.max()
    return int(counts.loc[counts == max_count].sort_index().index[0])


def _first_present(frame: pd.DataFrame, aliases: tuple[str, ...]) -> Any | None:
    lower_to_column = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        column = lower_to_column.get(alias.strip().lower())
        if column is not None:
            return column
    return None


def _probability_columns(frame: pd.DataFrame) -> list[tuple[int, Any]]:
    columns: list[tuple[int, Any]] = []
    lower_to_column = {str(column).strip().lower(): column for column in frame.columns}
    for class_id in VALID_CLASS_IDS:
        found = lower_to_column.get(str(class_id))
        if found is None:
            for prefix in PROBABILITY_PREFIXES:
                found = lower_to_column.get(f"{prefix}{class_id}".lower())
                if found is not None:
                    break
        if found is not None:
            columns.append((int(class_id), found))
    return columns


def _valid_probability_class_ids(class_ids: tuple[int, ...]) -> bool:
    return (
        len(class_ids) >= MIN_PROBABILITY_CLASS_COUNT
        and all(class_id in VALID_CLASS_IDS for class_id in class_ids)
    )


def _validate_class_series(values: pd.Series, *, name: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"{name} contains non-finite class labels")
    rounded = numeric.astype(int)
    if not np.allclose(numeric.to_numpy(float), rounded.to_numpy(float)):
        raise ValueError(f"{name} contains non-integer class labels")
    bad = sorted(set(rounded.loc[~rounded.isin(VALID_CLASS_IDS)].astype(int).tolist()))
    if bad:
        allowed = ", ".join(str(class_id) for class_id in VALID_CLASS_IDS)
        raise ValueError(f"{name} contains class labels outside {{{allowed}}}: {bad}")


def _finite_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else None


def _finite_min(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.min()) if not finite.empty else None


def _finite_max(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.max()) if not finite.empty else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
