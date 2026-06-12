"""Submission and metric-export helpers for MMUAD-style experiments."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = (
    "sequence_id",
    "time_s",
    "track_id",
    "x_m",
    "y_m",
    "z_m",
    "score",
)

UG2_RESULT_COLUMNS = (
    "sequence_id",
    "timestamp",
    "x",
    "y",
    "z",
    "uav_type",
    "score",
)

_SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "id",
    "name",
)
_UAV_TYPE_ALIASES = (
    "uav_type",
    "class_name",
    "class",
    "label",
    "category",
    "type",
    "uav_class",
)
_CLASS_MAP_KEYS = ("sequences", "class_map", "classes", "mapping", "items")
_CLASS_MAP_METADATA_KEYS = ("schema", "version", "description", "metadata")


def load_sequence_class_map(path: Path | None) -> dict[str, str]:
    """Load a sequence-to-UAV-type map from CSV or JSON."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _class_map_from_json_payload(payload)

    frame = pd.read_csv(path)
    lower = {str(col).lower(): col for col in frame.columns}
    rename = {}
    for alias in _SEQUENCE_ID_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "sequence_id"
            break
    for alias in _UAV_TYPE_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "uav_type"
            break
    frame = frame.rename(columns=rename)
    missing = {"sequence_id", "uav_type"}.difference(frame.columns)
    if missing:
        raise ValueError(f"class-map CSV missing columns: {sorted(missing)}")
    return {
        str(row["sequence_id"]): str(row["uav_type"])
        for _, row in frame.iterrows()
        if pd.notna(row["sequence_id"]) and pd.notna(row["uav_type"])
    }


def _class_map_from_json_payload(payload: Any) -> dict[str, str]:
    if isinstance(payload, list):
        class_map = _class_map_from_rows(payload)
        if class_map:
            return class_map
        raise ValueError("class-map JSON rows must contain sequence id and UAV type fields")
    if not isinstance(payload, dict):
        raise ValueError("class-map JSON must be an object or a list of sequence rows")

    for key in _CLASS_MAP_KEYS:
        nested = payload.get(key)
        class_map = _class_map_from_nested(nested)
        if class_map:
            return class_map

    class_map = _class_map_from_rows([payload])
    if class_map:
        return class_map
    class_map = _class_map_from_mapping(payload)
    if class_map:
        return class_map
    raise ValueError("class-map JSON does not contain any sequence UAV types")


def _class_map_from_nested(value: Any) -> dict[str, str]:
    if isinstance(value, list):
        return _class_map_from_rows(value)
    if isinstance(value, dict):
        return _class_map_from_mapping(value)
    return {}


def _class_map_from_rows(rows: list[Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sequence_id = _entry_value(row, _SEQUENCE_ID_ALIASES)
        uav_type = _entry_value(row, _UAV_TYPE_ALIASES)
        if sequence_id is not None and uav_type is not None:
            out[sequence_id] = uav_type
    return out


def _class_map_from_mapping(mapping: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if str(key).lower() in _CLASS_MAP_KEYS + _CLASS_MAP_METADATA_KEYS:
            continue
        sequence_id = _scalar_to_text(key)
        if sequence_id is None:
            continue
        if isinstance(value, dict):
            mapped_sequence_id = _entry_value(value, _SEQUENCE_ID_ALIASES)
            uav_type = _entry_value(value, _UAV_TYPE_ALIASES)
            if mapped_sequence_id is not None:
                sequence_id = mapped_sequence_id
        else:
            uav_type = _scalar_to_text(value)
        if uav_type is not None:
            out[sequence_id] = uav_type
    return out


def _entry_value(entry: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    lower_keys = {str(key).lower(): key for key in entry}
    for alias in aliases:
        key = alias if alias in entry else lower_keys.get(alias)
        if key is None:
            continue
        value = _scalar_to_text(entry[key])
        if value is not None:
            return value
    return None


def _scalar_to_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if not isinstance(value, str | int | float):
        return None
    text = str(value).strip()
    return text or None


def _estimate_sequence_values(
    estimates: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
) -> pd.Series:
    """Return one non-empty string sequence id per estimate row."""

    if "sequence_id" in estimates.columns:
        values = estimates["sequence_id"].fillna(default_sequence_id).astype(str).str.strip()
        return values.where(values != "", default_sequence_id)
    return pd.Series([default_sequence_id] * len(estimates), index=estimates.index)


def estimates_to_mmaud_results_frame(
    estimates: pd.DataFrame,
    *,
    class_name: str = "unknown",
    class_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Convert estimates into a Codabench-style ``mmaud_results.csv`` table.

    The public Codabench instructions require a ZIP containing a single file
    named ``mmaud_results.csv``.  The exact competition evaluator schema is not
    bundled with this repository, so this helper writes a compact, documented
    trajectory table that can be adapted once the official README/evaluator is
    available.
    """

    if estimates.empty:
        return pd.DataFrame(columns=UG2_RESULT_COLUMNS)
    sequence_values = _estimate_sequence_values(estimates)
    if "class_name" in estimates.columns:
        class_values = estimates["class_name"].fillna(class_name).astype(str)
    else:
        class_values = pd.Series([class_name] * len(estimates), index=estimates.index)
    if class_map:
        class_values = pd.Series(
            [
                class_map.get(str(seq), str(cls))
                for seq, cls in zip(sequence_values, class_values, strict=False)
            ],
            index=estimates.index,
        )
    frame = pd.DataFrame(
        {
            "sequence_id": sequence_values,
            "timestamp": estimates["time_s"].astype(float),
            "x": estimates["state_x_m"].astype(float),
            "y": estimates["state_y_m"].astype(float),
            "z": estimates["state_z_m"].astype(float),
            "uav_type": class_values,
            "score": 1.0,
        }
    )
    return frame[list(UG2_RESULT_COLUMNS)].sort_values(
        ["sequence_id", "timestamp"]
    ).reset_index(drop=True)


def write_mmaud_results_csv(
    estimates: pd.DataFrame,
    path: Path,
    *,
    class_name: str = "unknown",
    class_map: dict[str, str] | None = None,
) -> Path:
    """Write a Codabench-style ``mmaud_results.csv`` trajectory table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates_to_mmaud_results_frame(
        estimates, class_name=class_name, class_map=class_map
    ).to_csv(
        path, index=False
    )
    return path


def write_ug2_codabench_zip(
    estimates: pd.DataFrame,
    path: Path,
    *,
    class_name: str = "unknown",
    class_map: dict[str, str] | None = None,
) -> Path:
    """Write a UG2+ Codabench-style ZIP with exactly ``mmaud_results.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = estimates_to_mmaud_results_frame(
        estimates, class_name=class_name, class_map=class_map
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", frame.to_csv(index=False))
    return path


def inspect_submission_zip(path: Path) -> dict[str, Any]:
    """Return a small structural summary for a submission ZIP."""

    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        has_mmaud = "mmaud_results.csv" in names
        row_count = None
        columns: list[str] | None = None
        if has_mmaud:
            from io import BytesIO

            with archive.open("mmaud_results.csv") as handle:
                frame = pd.read_csv(BytesIO(handle.read()))
            row_count = int(len(frame))
            columns = list(frame.columns)
    return {
        "path": str(path),
        "members": names,
        "has_mmaud_results_csv": has_mmaud,
        "row_count": row_count,
        "columns": columns,
    }


def estimates_to_submission_frame(
    estimates: pd.DataFrame,
    *,
    track_id: str = "raft_uav_pp",
    use_estimate_track_ids: bool = True,
) -> pd.DataFrame:
    """Convert tracker estimates into a simple challenge-ready trajectory table."""

    if estimates.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
    numeric = pd.DataFrame(
        {
            "time_s": pd.to_numeric(estimates["time_s"], errors="coerce"),
            "x_m": pd.to_numeric(estimates["state_x_m"], errors="coerce"),
            "y_m": pd.to_numeric(estimates["state_y_m"], errors="coerce"),
            "z_m": pd.to_numeric(estimates["state_z_m"], errors="coerce"),
        },
        index=estimates.index,
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    work = estimates.loc[finite].copy()
    numeric = numeric.loc[finite]
    if work.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    track_values = pd.Series(str(track_id), index=work.index)
    if use_estimate_track_ids and "output_track_id" in estimates.columns:
        track_values = (
            work["output_track_id"]
            .where(work["output_track_id"].notna(), str(track_id))
            .astype(str)
        )
    rows = pd.DataFrame(
        {
            "sequence_id": _estimate_sequence_values(work),
            "time_s": numeric["time_s"],
            "track_id": track_values,
            "x_m": numeric["x_m"],
            "y_m": numeric["y_m"],
            "z_m": numeric["z_m"],
            "score": 1.0,
        }
    )
    return (
        rows[list(SUBMISSION_COLUMNS)]
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def write_submission_csv(
    estimates: pd.DataFrame,
    path: Path,
    *,
    track_id: str = "raft_uav_pp",
) -> Path:
    """Write a simple single-UAV trajectory submission CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates_to_submission_frame(estimates, track_id=track_id).to_csv(path, index=False)
    return path


def write_submission_json(
    estimates: pd.DataFrame,
    path: Path,
    *,
    track_id: str = "raft_uav_pp",
) -> Path:
    """Write a simple JSON trajectory export.

    This is not the official UG2+ upload schema; it is a stable interchange file
    for downstream conversion once the official evaluator/submission format is
    available.
    """

    frame = estimates_to_submission_frame(estimates, track_id=track_id)
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-single-uav-trajectory-v1",
        "track_id": track_id,
        "sequences": {},
    }
    for sequence_id, group in frame.groupby("sequence_id", sort=True):
        payload["sequences"][str(sequence_id)] = group.drop(
            columns=["sequence_id"]
        ).to_dict(orient="records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_submission_zip(
    estimates: pd.DataFrame,
    path: Path,
    *,
    track_id: str = "raft_uav_pp",
    include_json: bool = True,
) -> Path:
    """Write a portable ZIP bundle with CSV and optional JSON trajectory files."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = estimates_to_submission_frame(estimates, track_id=track_id)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission.csv", frame.to_csv(index=False))
        if include_json:
            payload: dict[str, Any] = {
                "schema": "raft-uav-mmuad-single-uav-trajectory-v1",
                "track_id": track_id,
                "sequences": {},
            }
            for sequence_id, group in frame.groupby("sequence_id", sort=True):
                payload["sequences"][str(sequence_id)] = group.drop(
                    columns=["sequence_id"]
                ).to_dict(orient="records")
            archive.writestr("submission.json", json.dumps(payload, indent=2))
    return path


def compute_trajectory_metrics(estimates: pd.DataFrame) -> dict[str, Any]:
    """Compute extra trajectory metrics when truth-error columns are present."""

    if estimates.empty or "error_3d_m" not in estimates.columns:
        return {"count": int(len(estimates))}
    rows: dict[str, Any] = {"sequences": {}, "pooled": _metrics_for_frame(estimates)}
    if "sequence_id" in estimates.columns:
        for sequence_id, group in estimates.groupby("sequence_id", sort=True):
            rows["sequences"][str(sequence_id)] = _metrics_for_frame(group)
    return rows


def _metrics_for_frame(frame: pd.DataFrame) -> dict[str, Any]:
    err = frame["error_3d_m"].to_numpy(float)
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return {"count": 0}
    out = {
        "count": int(finite.size),
        "mean_3d_m": float(np.mean(finite)),
        "rmse_3d_m": float(np.sqrt(np.mean(finite**2))),
        "p95_3d_m": float(np.percentile(finite, 95.0)),
        "max_3d_m": float(np.max(finite)),
        "ade_3d_m": float(np.mean(finite)),
        "fde_3d_m": _final_error(frame, "error_3d_m"),
    }
    if "error_2d_m" in frame.columns:
        err2 = frame["error_2d_m"].to_numpy(float)
        finite2 = err2[np.isfinite(err2)]
        if finite2.size:
            out.update(
                {
                    "mean_2d_m": float(np.mean(finite2)),
                    "p95_2d_m": float(np.percentile(finite2, 95.0)),
                    "max_2d_m": float(np.max(finite2)),
                    "ade_2d_m": float(np.mean(finite2)),
                    "fde_2d_m": _final_error(frame, "error_2d_m"),
                }
            )
    return out


def _final_error(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    if "time_s" not in frame.columns:
        return float(values[np.flatnonzero(finite)[-1]])
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    timed = finite & np.isfinite(times)
    if not timed.any():
        return float(values[np.flatnonzero(finite)[-1]])
    timed_indices = np.flatnonzero(timed)
    latest_time = float(np.max(times[timed_indices]))
    latest_indices = timed_indices[times[timed_indices] == latest_time]
    return float(values[latest_indices[-1]])
