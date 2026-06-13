"""Submission and metric-export helpers for MMUAD-style experiments."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_time_column_aliases


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

OFFICIAL_UG2_RESULT_COLUMNS = (
    "Sequence",
    "Timestamp",
    "Position",
    "Classification",
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
_COORDINATE_COLUMN_SETS = (
    ("state_x_m", "state_y_m", "state_z_m"),
    ("east_m", "north_m", "up_m"),
    ("x_m", "y_m", "z_m"),
    ("x", "y", "z"),
)


@dataclass(frozen=True)
class OfficialTrack5Validation:
    """Structural and timestamp-coverage validation for a Track 5 upload."""

    summary: dict[str, Any]
    rows: pd.DataFrame


def load_sequence_class_map(path: Path | None) -> dict[str, str]:
    """Load a sequence-to-UAV-type map from CSV, JSON, or YAML."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _load_class_map_payload(path)
        return _class_map_from_payload(payload)

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


def _load_class_map_payload(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return json.loads(text)
    return yaml.safe_load(text)


def _class_map_from_payload(payload: Any) -> dict[str, str]:
    if isinstance(payload, list):
        class_map = _class_map_from_rows(payload)
        if class_map:
            return class_map
        raise ValueError("class-map rows must contain sequence id and UAV type fields")
    if not isinstance(payload, dict):
        raise ValueError("class-map must be an object or a list of sequence rows")

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
    raise ValueError("class-map does not contain any sequence UAV types")


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


def _estimate_track_id_values(
    estimates: pd.DataFrame,
    *,
    track_id: str,
    use_estimate_track_ids: bool,
) -> pd.Series:
    """Return one non-empty track id per estimate row."""

    default_track_id = str(track_id)
    if use_estimate_track_ids and "output_track_id" in estimates.columns:
        values = (
            estimates["output_track_id"]
            .where(estimates["output_track_id"].notna(), default_track_id)
            .astype(str)
            .str.strip()
        )
        missing = values.eq("") | values.str.lower().isin({"nan", "none", "<na>"})
        return values.where(~missing, default_track_id)
    return pd.Series([default_track_id] * len(estimates), index=estimates.index)


def _estimate_coordinate_columns(estimates: pd.DataFrame) -> tuple[str, str, str]:
    """Return x/y/z coordinate columns for supported estimate table schemas."""

    for columns in _COORDINATE_COLUMN_SETS:
        if all(column in estimates.columns for column in columns):
            return columns
    expected = " or ".join(", ".join(columns) for columns in _COORDINATE_COLUMN_SETS)
    raise KeyError(f"estimates must contain coordinate columns: {expected}")


def _estimate_time_values(estimates: pd.DataFrame) -> pd.Series:
    """Return numeric estimate timestamps from tracker or result-table columns."""

    if "time_s" in estimates.columns:
        return pd.to_numeric(estimates["time_s"], errors="coerce")
    if "timestamp" in estimates.columns:
        return pd.to_numeric(estimates["timestamp"], errors="coerce")
    raise KeyError("estimates must contain a 'time_s' or 'timestamp' column")


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
    x_column, y_column, z_column = _estimate_coordinate_columns(estimates)
    time_values = _estimate_time_values(estimates)
    sequence_values = _estimate_sequence_values(estimates)
    if "class_name" in estimates.columns:
        class_values = estimates["class_name"].fillna(class_name).astype(str)
    elif "uav_type" in estimates.columns:
        class_values = estimates["uav_type"].fillna(class_name).astype(str)
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
            "timestamp": time_values.astype(float),
            "x": estimates[x_column].astype(float),
            "y": estimates[y_column].astype(float),
            "z": estimates[z_column].astype(float),
            "uav_type": class_values,
            "score": 1.0,
        }
    )
    return frame[list(UG2_RESULT_COLUMNS)].sort_values(
        ["sequence_id", "timestamp"]
    ).reset_index(drop=True)


def estimates_to_official_mmaud_results_frame(
    estimates: pd.DataFrame,
    *,
    classification: int | str = 0,
    class_map: dict[str, str] | None = None,
    invalid_row_policy: Literal["raise", "drop"] = "raise",
) -> pd.DataFrame:
    """Convert estimates into the public Track 5 ``mmaud_results.csv`` schema.

    The CVPR 2024 UG2+ Track 5 README specifies columns named ``Sequence``,
    ``Timestamp``, ``Position``, and ``Classification``.  ``Position`` is written
    as a compact ``(x,y,z)`` string because CSV has no native NumPy-array type.
    ``Classification`` must be an integer; pass a sequence class map with
    numeric values when per-sequence labels are known.
    """

    policy = _normalize_official_invalid_row_policy(invalid_row_policy)
    if estimates.empty:
        return pd.DataFrame(columns=OFFICIAL_UG2_RESULT_COLUMNS)
    x_column, y_column, z_column = _estimate_coordinate_columns(estimates)
    time_values = _estimate_time_values(estimates)
    sequence_values = _estimate_sequence_values(estimates)
    classification_values = _estimate_classification_values(
        estimates,
        sequence_values=sequence_values,
        default_classification=classification,
        class_map=class_map,
    )
    numeric = pd.DataFrame(
        {
            "Timestamp": time_values,
            "x": pd.to_numeric(estimates[x_column], errors="coerce"),
            "y": pd.to_numeric(estimates[y_column], errors="coerce"),
            "z": pd.to_numeric(estimates[z_column], errors="coerce"),
            "Classification": classification_values,
        },
        index=estimates.index,
    )
    finite = np.isfinite(numeric[["Timestamp", "x", "y", "z"]].to_numpy(dtype=float)).all(axis=1)
    if not finite.all() and policy == "raise":
        _raise_nonfinite_official_estimate_rows(numeric, finite)
    work_sequences = sequence_values.loc[finite]
    numeric = numeric.loc[finite]
    rows = pd.DataFrame(
        {
            "Sequence": work_sequences.astype(str),
            "Timestamp": numeric["Timestamp"].astype(float),
            "Position": [
                _format_official_position(x, y, z)
                for x, y, z in zip(
                    numeric["x"],
                    numeric["y"],
                    numeric["z"],
                    strict=False,
                )
            ],
            "Classification": numeric["Classification"].astype(int),
        }
    )
    return rows[list(OFFICIAL_UG2_RESULT_COLUMNS)].sort_values(
        ["Sequence", "Timestamp"]
    ).reset_index(drop=True)


def _normalize_official_invalid_row_policy(value: str) -> Literal["raise", "drop"]:
    policy = str(value).strip().lower()
    if policy not in {"raise", "drop"}:
        raise ValueError("invalid_row_policy must be 'raise' or 'drop'")
    return policy  # type: ignore[return-value]


def _raise_nonfinite_official_estimate_rows(
    numeric: pd.DataFrame,
    finite: np.ndarray,
) -> None:
    columns = ("Timestamp", "x", "y", "z")
    examples: list[str] = []
    invalid = numeric.loc[~finite, list(columns)].head(5)
    for row_index, row in invalid.iterrows():
        bad_columns = [
            column
            for column in columns
            if not np.isfinite(float(row[column]))
        ]
        examples.append(f"{row_index}:{','.join(bad_columns)}")
    remaining = int((~finite).sum()) - len(examples)
    suffix = f"; {remaining} more" if remaining > 0 else ""
    raise ValueError(
        "official Track 5 output contains non-finite Timestamp/Position "
        "estimate rows; fix the tracker output or pass invalid_row_policy='drop' "
        f"for a diagnostic export only. invalid_rows={examples}{suffix}"
    )


def _estimate_classification_values(
    estimates: pd.DataFrame,
    *,
    sequence_values: pd.Series,
    default_classification: int | str,
    class_map: dict[str, str] | None,
) -> pd.Series:
    if "classification" in estimates.columns:
        raw = estimates["classification"]
    elif "class_id" in estimates.columns:
        raw = estimates["class_id"]
    elif "class_name" in estimates.columns:
        raw = estimates["class_name"]
    elif "uav_type" in estimates.columns:
        raw = estimates["uav_type"]
    else:
        raw = pd.Series([default_classification] * len(estimates), index=estimates.index)
    values = pd.Series(raw, index=estimates.index).copy()
    if class_map:
        mapped = [
            class_map.get(str(sequence_id), value)
            for sequence_id, value in zip(sequence_values, values, strict=False)
        ]
        values = pd.Series(mapped, index=estimates.index)
    return values.map(_classification_to_int)


def parse_official_classification_cell(value: Any) -> int:
    """Parse a public Track 5 ``Classification`` cell into an integer id."""

    if value is None:
        raise ValueError("official MMUAD Classification values must be integers")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("official MMUAD Classification values must be integer ids, not booleans")
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        raise ValueError("official MMUAD Classification values must be integers")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("official MMUAD Classification values must be integers")
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(
                "official MMUAD Classification values must be integer ids; "
                f"got {value!r}"
            ) from exc
    else:
        number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {value!r}"
        )
    return int(number)


def _classification_to_int(value: Any) -> int:
    return parse_official_classification_cell(value)


def _format_official_position(x: Any, y: Any, z: Any) -> str:
    return f"({_format_float(x)},{_format_float(y)},{_format_float(z)})"


def _format_float(value: Any) -> str:
    return f"{float(value):.12g}"


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


def write_official_mmaud_results_csv(
    estimates: pd.DataFrame,
    path: Path,
    *,
    classification: int | str = 0,
    class_map: dict[str, str] | None = None,
    invalid_row_policy: Literal["raise", "drop"] = "raise",
) -> Path:
    """Write the public CVPR 2024 UG2+ Track 5 ``mmaud_results.csv`` schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    estimates_to_official_mmaud_results_frame(
        estimates,
        classification=classification,
        class_map=class_map,
        invalid_row_policy=invalid_row_policy,
    ).to_csv(path, index=False)
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


def write_official_ug2_codabench_zip(
    estimates: pd.DataFrame,
    path: Path,
    *,
    classification: int | str = 0,
    class_map: dict[str, str] | None = None,
    invalid_row_policy: Literal["raise", "drop"] = "raise",
) -> Path:
    """Write a Track 5 upload ZIP containing only ``mmaud_results.csv``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = estimates_to_official_mmaud_results_frame(
        estimates,
        classification=classification,
        class_map=class_map,
        invalid_row_policy=invalid_row_policy,
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


def validate_official_track5_submission(
    path: Path,
    *,
    template: pd.DataFrame | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> OfficialTrack5Validation:
    """Validate the public UG2+ Track 5 upload structure.

    This is a local preflight check for leaderboard packaging.  It enforces the
    public ZIP/CSV schema and, when a timestamp template is supplied, checks
    whether each requested sequence timestamp has exactly one prediction.
    """

    if timestamp_tolerance_s < 0.0:
        raise ValueError("timestamp_tolerance_s must be non-negative")
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    members: list[str] = []
    frame: pd.DataFrame | None = None
    is_zip = path.suffix.lower() == ".zip"
    if require_zip and not is_zip:
        errors.append("official Track 5 Codabench upload must be a .zip file")
    try:
        if is_zip:
            frame, members = _read_official_track5_zip_for_validation(path, errors)
        else:
            frame = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"could not read official Track 5 submission: {exc}")

    row_diagnostics = pd.DataFrame()
    schema_summary: dict[str, Any] = {}
    if frame is not None:
        schema_summary, row_diagnostics = _validate_official_track5_frame(
            frame,
            template=template,
            timestamp_tolerance_s=timestamp_tolerance_s,
            errors=errors,
            warnings=warnings,
        )
    template_checked = template is not None
    summary: dict[str, Any] = {
        "schema": "raft-uav-mmuad-official-track5-validation-v1",
        "path": str(path),
        "is_zip": bool(is_zip),
        "require_zip": bool(require_zip),
        "members": members,
        "has_mmaud_results_csv": "mmaud_results.csv" in members if is_zip else path.exists(),
        "contains_only_mmaud_results_csv": members == ["mmaud_results.csv"] if is_zip else False,
        "expected_columns": list(OFFICIAL_UG2_RESULT_COLUMNS),
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
        "template_checked": bool(template_checked),
        "errors": errors,
        "warnings": warnings,
        **schema_summary,
    }
    summary["valid"] = bool(
        not errors
        and int(summary.get("invalid_sequence_count", 0)) == 0
        and int(summary.get("invalid_timestamp_count", 0)) == 0
        and int(summary.get("invalid_position_count", 0)) == 0
        and int(summary.get("invalid_classification_count", 0)) == 0
        and int(summary.get("duplicate_prediction_count", 0)) == 0
        and (not template_checked or int(summary.get("missing_template_timestamp_count", 0)) == 0)
        and (not template_checked or int(summary.get("extra_prediction_count", 0)) == 0)
    )
    return OfficialTrack5Validation(summary=summary, rows=row_diagnostics)


def load_official_track5_template_file(path: Path) -> pd.DataFrame:
    """Load requested Track 5 ``Sequence``/``Timestamp`` rows from CSV or ZIP.

    The public upload schema and sample/template files may already use
    ``Sequence,Timestamp,Position,Classification``.  Validation only needs the
    requested sequence timestamps, so this helper accepts official result-like
    files and returns the normalized template columns used by the preflight
    checker.
    """

    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            if "mmaud_results.csv" not in archive.namelist():
                raise ValueError(f"{path} does not contain 'mmaud_results.csv'")
            with archive.open("mmaud_results.csv") as handle:
                frame = pd.read_csv(BytesIO(handle.read()))
    else:
        frame = pd.read_csv(path)
    return _normalize_track5_template(frame)


def _read_official_track5_zip_for_validation(
    path: Path,
    errors: list[str],
) -> tuple[pd.DataFrame | None, list[str]]:
    with ZipFile(path) as archive:
        members = archive.namelist()
        if members != ["mmaud_results.csv"]:
            errors.append("official Track 5 ZIP must contain only mmaud_results.csv")
        if "mmaud_results.csv" not in members:
            return None, members
        with archive.open("mmaud_results.csv") as handle:
            frame = pd.read_csv(BytesIO(handle.read()))
    return frame, members


def _validate_official_track5_frame(
    frame: pd.DataFrame,
    *,
    template: pd.DataFrame | None,
    timestamp_tolerance_s: float,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = list(frame.columns)
    if columns != list(OFFICIAL_UG2_RESULT_COLUMNS):
        errors.append(
            "mmaud_results.csv columns must exactly equal "
            f"{list(OFFICIAL_UG2_RESULT_COLUMNS)}"
        )
    diagnostics, normalized = _official_track5_row_diagnostics(frame)
    invalid_sequence_count = int((diagnostics["status"] == "invalid_sequence").sum())
    invalid_timestamp_count = int((diagnostics["status"] == "invalid_timestamp").sum())
    invalid_position_count = int((diagnostics["status"] == "invalid_position").sum())
    invalid_classification_count = int(
        (diagnostics["status"] == "invalid_classification").sum()
    )
    duplicate_count = 0
    extra_count = 0
    missing_count = 0
    template_count = None
    if not normalized.empty:
        duplicate_indices = _duplicate_prediction_indices(
            normalized,
            timestamp_tolerance_s=timestamp_tolerance_s,
        )
        duplicate_count = len(duplicate_indices)
        if duplicate_indices:
            diagnostics.loc[
                diagnostics["row_index"].isin(duplicate_indices)
                & diagnostics["status"].eq("ok"),
                "status",
            ] = "duplicate_prediction"
    if template is not None:
        try:
            template_rows = _normalize_track5_template(template)
        except ValueError as exc:
            errors.append(str(exc))
            template_rows = pd.DataFrame(columns=["sequence_id", "time_s"])
        template_count = int(len(template_rows))
        if not template_rows.empty:
            coverage = _track5_template_coverage_rows(
                normalized,
                template_rows,
                timestamp_tolerance_s=timestamp_tolerance_s,
            )
            missing_count = int((coverage["status"] == "missing_template_timestamp").sum())
            extra_indices = set(
                coverage.loc[coverage["status"] == "extra_prediction", "row_index"]
                .dropna()
                .astype(int)
                .tolist()
            )
            extra_count = len(extra_indices)
            if extra_indices:
                diagnostics.loc[
                    diagnostics["row_index"].isin(extra_indices)
                    & diagnostics["status"].eq("ok"),
                    "status",
                ] = "extra_prediction"
            diagnostics = pd.concat([diagnostics, coverage], ignore_index=True, sort=False)
    valid_row_count = int((diagnostics["status"] == "ok").sum())
    summary = {
        "columns": columns,
        "row_count": int(len(frame)),
        "valid_row_count": valid_row_count,
        "invalid_sequence_count": invalid_sequence_count,
        "invalid_timestamp_count": invalid_timestamp_count,
        "invalid_position_count": invalid_position_count,
        "invalid_classification_count": invalid_classification_count,
        "duplicate_prediction_count": int(duplicate_count),
        "template_timestamp_count": template_count,
        "missing_template_timestamp_count": int(missing_count),
        "extra_prediction_count": int(extra_count),
    }
    return summary, diagnostics


def _official_track5_row_diagnostics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for row_index, row in frame.iterrows():
        sequence = _official_sequence_text(row.get("Sequence"))
        timestamp = np.nan
        status = "ok"
        reason = ""
        xyz: tuple[float, float, float] | None = None
        classification: int | None = None
        if sequence is None:
            sequence = ""
            status = "invalid_sequence"
            reason = "blank or missing Sequence"
        if status == "ok":
            try:
                timestamp = parse_official_timestamp_cell(row.get("Timestamp"))
            except ValueError as exc:
                status = "invalid_timestamp"
                reason = str(exc)
        if status == "ok":
            try:
                xyz = parse_official_position_cell(row.get("Position"))
            except ValueError as exc:
                status = "invalid_position"
                reason = str(exc)
        if status == "ok":
            try:
                classification = parse_official_classification_cell(row.get("Classification"))
            except ValueError as exc:
                status = "invalid_classification"
                reason = str(exc)
        record = {
            "row_type": "prediction",
            "row_index": int(row_index),
            "sequence_id": sequence,
            "timestamp": float(timestamp) if np.isfinite(float(timestamp)) else np.nan,
            "status": status,
            "reason": reason,
            "classification": classification,
        }
        if xyz is not None:
            record.update({"x": xyz[0], "y": xyz[1], "z": xyz[2]})
        rows.append(record)
        if status == "ok" and xyz is not None and classification is not None:
            normalized_rows.append(
                {
                    "row_index": int(row_index),
                    "sequence_id": sequence,
                    "timestamp": float(timestamp),
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    "classification": classification,
                }
            )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(normalized_rows)


def parse_official_position_cell(value: Any) -> tuple[float, float, float]:
    """Parse a public Track 5 ``Position`` cell into finite ``x,y,z`` floats."""

    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = _split_official_position_text(text)
    else:
        parsed = value
    if isinstance(parsed, np.ndarray):
        values = parsed.reshape(-1).tolist()
    elif isinstance(parsed, list | tuple):
        values = list(parsed)
    else:
        raise ValueError(f"invalid Track 5 Position value: {value!r}")
    if len(values) != 3:
        raise ValueError(f"Track 5 Position must contain exactly 3 values: {value!r}")
    xyz = (float(values[0]), float(values[1]), float(values[2]))
    if not np.isfinite(np.asarray(xyz, dtype=float)).all():
        raise ValueError(f"Track 5 Position must contain finite values: {value!r}")
    return xyz


def parse_official_sequence_cell(value: Any) -> str:
    """Parse a public Track 5 ``Sequence`` cell into a non-missing id."""

    text = _scalar_to_text(value)
    if text is None or text.lower() in {"nan", "none", "<na>"}:
        raise ValueError("official MMUAD Sequence values must be nonblank")
    return text


def parse_official_timestamp_cell(value: Any) -> float:
    """Parse a public Track 5 ``Timestamp`` cell into a finite float."""

    if value is None:
        raise ValueError("official MMUAD Timestamp values must be finite numbers")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("official MMUAD Timestamp values must be finite numbers, not booleans")
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        raise ValueError("official MMUAD Timestamp values must be finite numbers")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "official MMUAD Timestamp values must be finite numbers; "
            f"got {value!r}"
        ) from exc
    if not np.isfinite(timestamp):
        raise ValueError(
            "official MMUAD Timestamp values must be finite numbers; "
            f"got {value!r}"
        )
    return timestamp


def _official_sequence_text(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _split_official_position_text(text: str) -> list[str]:
    stripped = _strip_official_position_array_wrapper(text).strip("[]()")
    separator = "," if "," in stripped else None
    if ";" in stripped:
        stripped = stripped.replace(";", "," if separator == "," else " ")
    if separator == ",":
        return [part for part in stripped.split(",") if part.strip()]
    return [part for part in stripped.split() if part.strip()]


def _strip_official_position_array_wrapper(text: str) -> str:
    stripped = text.strip()
    lower = stripped.lower()
    for prefix in ("array(", "np.array(", "numpy.array("):
        if not lower.startswith(prefix):
            continue
        inner = stripped[len(prefix) :].strip()
        if inner.endswith(")"):
            inner = inner[:-1].strip()
        bracketed = _leading_balanced_group(inner, opener="[", closer="]")
        if bracketed is not None:
            return bracketed
        parenthesized = _leading_balanced_group(inner, opener="(", closer=")")
        if parenthesized is not None:
            return parenthesized
        dtype_start = inner.lower().find(", dtype=")
        if dtype_start >= 0:
            return inner[:dtype_start].strip()
        return inner
    return stripped


def _leading_balanced_group(text: str, *, opener: str, closer: str) -> str | None:
    if not text.startswith(opener):
        return None
    depth = 0
    for index, char in enumerate(text):
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[: index + 1]
    return None


def _duplicate_prediction_indices(
    rows: pd.DataFrame,
    *,
    timestamp_tolerance_s: float,
) -> set[int]:
    duplicates: set[int] = set()
    for _sequence_id, group in rows.groupby("sequence_id", sort=True):
        group = group.sort_values(["timestamp", "row_index"])
        last_time: float | None = None
        for _, row in group.iterrows():
            timestamp = float(row["timestamp"])
            if last_time is not None and abs(timestamp - last_time) <= timestamp_tolerance_s:
                duplicates.add(int(row["row_index"]))
            else:
                last_time = timestamp
    return duplicates


def _normalize_track5_template(template: pd.DataFrame) -> pd.DataFrame:
    rows = normalize_time_column_aliases(pd.DataFrame(template).copy(), target="time_s")
    lower_to_original = {str(column).lower(): column for column in rows.columns}
    rename = {}
    for alias in _SEQUENCE_ID_ALIASES:
        if alias in lower_to_original:
            rename[lower_to_original[alias]] = "sequence_id"
            break
    if "sequence" in lower_to_original and "sequence_id" not in rename.values():
        rename[lower_to_original["sequence"]] = "sequence_id"
    if "timestamp" in lower_to_original and "time_s" not in rows.columns:
        rename[lower_to_original["timestamp"]] = "time_s"
    rows = rows.rename(columns=rename)
    missing = {"sequence_id", "time_s"}.difference(rows.columns)
    if missing:
        raise ValueError(f"official Track 5 template missing columns: {sorted(missing)}")
    rows = rows[["sequence_id", "time_s"]].copy()
    rows["sequence_id"] = rows["sequence_id"].map(_official_sequence_text)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    finite = rows["sequence_id"].notna() & np.isfinite(rows["time_s"].to_numpy(float))
    rows = rows.loc[finite].drop_duplicates().sort_values(
        ["sequence_id", "time_s"]
    ).reset_index(drop=True)
    return rows


def _track5_template_coverage_rows(
    predictions: pd.DataFrame,
    template: pd.DataFrame,
    *,
    timestamp_tolerance_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matched_prediction_indices: set[int] = set()
    for sequence_id, group in template.groupby("sequence_id", sort=True):
        seq_predictions = predictions.loc[predictions["sequence_id"] == str(sequence_id)]
        prediction_times = seq_predictions["timestamp"].to_numpy(float)
        for _, template_row in group.iterrows():
            timestamp = float(template_row["time_s"])
            matched_rows = seq_predictions.loc[
                np.abs(prediction_times - timestamp) <= timestamp_tolerance_s
            ]
            if matched_rows.empty:
                rows.append(
                    {
                        "row_type": "template",
                        "row_index": np.nan,
                        "sequence_id": str(sequence_id),
                        "timestamp": timestamp,
                        "status": "missing_template_timestamp",
                        "reason": "no prediction at requested timestamp",
                    }
                )
                continue
            matched_prediction_indices.update(matched_rows["row_index"].astype(int).tolist())
            rows.append(
                {
                    "row_type": "template",
                    "row_index": int(matched_rows.iloc[0]["row_index"]),
                    "sequence_id": str(sequence_id),
                    "timestamp": timestamp,
                    "status": "covered_template_timestamp",
                    "reason": "",
                }
            )
    for _, prediction in predictions.iterrows():
        row_index = int(prediction["row_index"])
        if row_index in matched_prediction_indices:
            continue
        rows.append(
            {
                "row_type": "prediction",
                "row_index": row_index,
                "sequence_id": str(prediction["sequence_id"]),
                "timestamp": float(prediction["timestamp"]),
                "status": "extra_prediction",
                "reason": "prediction does not match a requested template timestamp",
            }
        )
    return pd.DataFrame.from_records(rows)


def estimates_to_submission_frame(
    estimates: pd.DataFrame,
    *,
    track_id: str = "raft_uav_pp",
    use_estimate_track_ids: bool = True,
) -> pd.DataFrame:
    """Convert tracker estimates into a simple challenge-ready trajectory table."""

    if estimates.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)
    x_column, y_column, z_column = _estimate_coordinate_columns(estimates)
    time_values = _estimate_time_values(estimates)
    numeric = pd.DataFrame(
        {
            "time_s": time_values,
            "x_m": pd.to_numeric(estimates[x_column], errors="coerce"),
            "y_m": pd.to_numeric(estimates[y_column], errors="coerce"),
            "z_m": pd.to_numeric(estimates[z_column], errors="coerce"),
        },
        index=estimates.index,
    )
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    work = estimates.loc[finite].copy()
    numeric = numeric.loc[finite]
    if work.empty:
        return pd.DataFrame(columns=SUBMISSION_COLUMNS)

    track_values = _estimate_track_id_values(
        work,
        track_id=track_id,
        use_estimate_track_ids=use_estimate_track_ids,
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
