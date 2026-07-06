#!/usr/bin/env python
"""Build a sequence class map from MMUAD prediction outputs.

The Track 5 packaging tools accept a ``sequence_id,uav_type`` class map.  This
helper turns dense tracker/classifier predictions or official-style
``mmaud_results.csv`` files into that class map without using pose or class
truth.  It is useful before template resampling when the pose output already
contains useful sequence labels that should not be overwritten by the default
classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.submission import (  # noqa: E402
    OFFICIAL_TRACK5_CLASS_IDS,
    load_official_track5_results_frame,
    parse_official_classification_cell,
    parse_official_sequence_cell,
)

SEQUENCE_ALIASES = ("Sequence", "sequence_id", "sequence", "seq", "scene", "scene_id")
CLASS_ALIASES = (
    "Classification",
    "classification",
    "class_id",
    "uav_type",
    "class",
    "label",
    "predicted_class",
    "predicted_class_id",
)
CONFIDENCE_ALIASES = (
    "classification_confidence",
    "sequence_class_confidence",
    "class_confidence",
    "confidence",
    "probability",
    "score",
)
TIME_ALIASES = ("Timestamp", "timestamp", "time_s", "timestamp_s", "time")
POLICIES = ("mode", "confidence", "last")
ClassMapPolicy = Literal["mode", "confidence", "last"]


def build_sequence_class_map_from_predictions(
    predictions: pd.DataFrame,
    *,
    policy: ClassMapPolicy = "mode",
    allow_non_track5_class_ids: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(class_map, per_sequence_diagnostics)`` from prediction rows."""

    normalized_policy = _normalize_policy(policy)
    rows = _normalize_prediction_rows(
        predictions,
        allow_non_track5_class_ids=allow_non_track5_class_ids,
    )
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        selected = _select_sequence_class(group, policy=normalized_policy)
        class_counts = group["classification"].value_counts().sort_index()
        class_means = group.groupby("classification", sort=True)["confidence"].mean()
        record = {
            "sequence_id": str(sequence_id),
            "uav_type": int(selected["classification"]),
        }
        records.append(record)
        diagnostics.append(
            {
                "sequence_id": str(sequence_id),
                "uav_type": int(selected["classification"]),
                "policy": normalized_policy,
                "row_count": int(len(group)),
                "selected_class_count": int(class_counts.get(selected["classification"], 0)),
                "selected_class_fraction": float(
                    class_counts.get(selected["classification"], 0) / max(len(group), 1)
                ),
                "selected_class_mean_confidence": _finite_float(
                    class_means.get(selected["classification"], np.nan),
                ),
                "class_count_0": int(class_counts.get(0, 0)),
                "class_count_1": int(class_counts.get(1, 0)),
                "class_count_2": int(class_counts.get(2, 0)),
                "class_count_3": int(class_counts.get(3, 0)),
                "mean_confidence": _finite_float(group["confidence"].mean()),
                "time_min_s": _finite_float(group["time_s"].min()),
                "time_max_s": _finite_float(group["time_s"].max()),
            }
        )
    class_map = pd.DataFrame.from_records(records, columns=["sequence_id", "uav_type"])
    diagnostics_frame = pd.DataFrame.from_records(diagnostics)
    return class_map, diagnostics_frame


def write_sequence_class_map_artifacts(
    *,
    predictions: pd.DataFrame,
    output_csv: Path,
    diagnostics_csv: Path | None = None,
    summary_json: Path | None = None,
    policy: ClassMapPolicy = "mode",
    allow_non_track5_class_ids: bool = False,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write sequence class map artifacts and return written paths."""

    class_map, diagnostics = build_sequence_class_map_from_predictions(
        predictions,
        policy=policy,
        allow_non_track5_class_ids=allow_non_track5_class_ids,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    class_map.to_csv(output_csv, index=False)
    paths = {"class_map_csv": output_csv}
    if diagnostics_csv is not None:
        diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(diagnostics_csv, index=False)
        paths["diagnostics_csv"] = diagnostics_csv
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema": "raft-uav-mmuad-sequence-class-map-from-predictions-v1",
            "policy": _normalize_policy(policy),
            "sequence_count": int(len(class_map)),
            "prediction_row_count": int(len(predictions)),
            "class_histogram": {
                str(class_id): int((class_map["uav_type"] == class_id).sum())
                for class_id in sorted(class_map["uav_type"].unique())
            },
            "allow_non_track5_class_ids": bool(allow_non_track5_class_ids),
            "paths": {name: str(path) for name, path in paths.items()},
            "provenance": provenance or {},
        }
        summary_json.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        paths["summary_json"] = summary_json
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="prediction CSV or ZIP")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="mode")
    parser.add_argument("--allow-non-track5-class-ids", action="store_true")
    args = parser.parse_args(argv)

    predictions = load_prediction_class_rows(args.predictions)
    paths = write_sequence_class_map_artifacts(
        predictions=predictions,
        output_csv=args.output_csv,
        diagnostics_csv=args.diagnostics_csv,
        summary_json=args.summary_json,
        policy=args.policy,
        allow_non_track5_class_ids=args.allow_non_track5_class_ids,
        provenance={"predictions": str(args.predictions)},
    )
    print("mmuad_sequence_class_map_from_predictions=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def load_prediction_class_rows(path: Path) -> pd.DataFrame:
    """Load class-bearing prediction rows from official or generic files."""

    path = Path(path)
    try:
        official = load_official_track5_results_frame(path)
    except Exception:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return official.rename(columns={"Sequence": "sequence_id", "Classification": "classification"})


def _normalize_prediction_rows(
    predictions: pd.DataFrame,
    *,
    allow_non_track5_class_ids: bool,
) -> pd.DataFrame:
    rows = pd.DataFrame(predictions).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    class_column = _first_present(rows, CLASS_ALIASES)
    if sequence_column is None or class_column is None:
        raise ValueError("predictions must contain sequence and classification columns")
    confidence_column = _first_present(rows, CONFIDENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    confidence = (
        pd.to_numeric(rows[confidence_column], errors="coerce")
        if confidence_column is not None
        else pd.Series(np.ones(len(rows), dtype=float), index=rows.index)
    )
    time_s = (
        pd.to_numeric(rows[time_column], errors="coerce")
        if time_column is not None
        else pd.Series(np.nan, index=rows.index, dtype=float)
    )
    out = pd.DataFrame(
        {
            "sequence_id": [_parse_sequence_id(value) for value in rows[sequence_column]],
            "classification": [
                _parse_class_id(value, allow_non_track5_class_ids=allow_non_track5_class_ids)
                for value in rows[class_column]
            ],
            "confidence": confidence.fillna(1.0).astype(float),
            "time_s": time_s.astype(float),
        }
    )
    valid = out["sequence_id"].ne("")
    valid &= out["sequence_id"].str.lower().ne("nan")
    return out.loc[valid].reset_index(drop=True)


def _select_sequence_class(group: pd.DataFrame, *, policy: ClassMapPolicy) -> pd.Series:
    if policy == "last":
        timed = group.sort_values("time_s", na_position="first")
        return timed.iloc[-1]
    if policy == "confidence":
        by_class = group.groupby("classification", sort=True).agg(
            mean_confidence=("confidence", "mean"),
            count=("classification", "size"),
        )
        by_class = by_class.sort_values(
            ["mean_confidence", "count"],
            ascending=[False, False],
        )
        class_id = int(by_class.index[0])
        return group.loc[group["classification"] == class_id].iloc[0]
    counts = group["classification"].value_counts().sort_index()
    max_count = int(counts.max())
    tied_classes = set(counts.loc[counts == max_count].index.astype(int))
    if len(tied_classes) == 1:
        class_id = int(next(iter(tied_classes)))
    else:
        tied = group.loc[group["classification"].isin(tied_classes)]
        means = tied.groupby("classification", sort=True)["confidence"].mean()
        class_id = int(means.sort_values(ascending=False).index[0])
    return group.loc[group["classification"] == class_id].iloc[0]


def _parse_sequence_id(value: Any) -> str:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return ""


def _parse_class_id(value: Any, *, allow_non_track5_class_ids: bool) -> int:
    class_id = parse_official_classification_cell(value)
    if not allow_non_track5_class_ids and class_id not in OFFICIAL_TRACK5_CLASS_IDS:
        allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(f"classification must be one of {{{allowed}}}; got {class_id!r}")
    return int(class_id)


def _first_present(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lower = {str(column).strip().lower(): column for column in rows.columns}
    for alias in aliases:
        if alias in rows.columns:
            return alias
        key = lower.get(alias.lower())
        if key is not None:
            return key
    return None


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _normalize_policy(policy: str) -> ClassMapPolicy:
    text = str(policy).strip().lower()
    if text not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    return text  # type: ignore[return-value]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
