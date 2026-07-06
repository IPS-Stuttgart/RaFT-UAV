"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raft_uav.mmuad import _submission_impl as _impl

_ORIGINAL_PARSE_ATTR = "_raft_uav_original_parse_official_classification_cell"
_ORIGINAL_LOAD_ATTR = "_raft_uav_original_load_sequence_class_map"
_ORIGINAL_NORMALIZE_ATTR = "_raft_uav_original_normalize_official_track5_results_frame"
_ORIGINAL_ROW_DIAGNOSTICS_ATTR = "_raft_uav_original_official_track5_row_diagnostics"

if not hasattr(_impl, _ORIGINAL_PARSE_ATTR):
    setattr(_impl, _ORIGINAL_PARSE_ATTR, _impl.parse_official_classification_cell)
if not hasattr(_impl, _ORIGINAL_LOAD_ATTR):
    setattr(_impl, _ORIGINAL_LOAD_ATTR, _impl.load_sequence_class_map)
if not hasattr(_impl, _ORIGINAL_NORMALIZE_ATTR):
    setattr(_impl, _ORIGINAL_NORMALIZE_ATTR, _impl.normalize_official_track5_results_frame)
if not hasattr(_impl, _ORIGINAL_ROW_DIAGNOSTICS_ATTR):
    setattr(_impl, _ORIGINAL_ROW_DIAGNOSTICS_ATTR, _impl._official_track5_row_diagnostics)

_parse_original = getattr(_impl, _ORIGINAL_PARSE_ATTR)
setattr(_impl, "_raft_uav_permissive_parse_official_classification_cell", _parse_original)
_load_sequence_class_map_original = getattr(_impl, _ORIGINAL_LOAD_ATTR)
_normalize_official_track5_results_frame_original = getattr(_impl, _ORIGINAL_NORMALIZE_ATTR)
_official_track5_row_diagnostics_original = getattr(_impl, _ORIGINAL_ROW_DIAGNOSTICS_ATTR)


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not booleans"
        )
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        allowed = ", ".join(str(item) for item in sorted(_impl.OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official Track 5 classification must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return class_id


def _load_sequence_class_map_with_official_sequences(path: Path | str | None) -> dict[str, str]:
    """Load class maps while canonicalizing CSV/JSON/YAML sequence ids like official rows."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return _canonicalize_sequence_class_map(_load_sequence_class_map_original(path))

    try:
        frame = _impl.pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        frame = _impl.pd.read_csv(path)
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[Any, str] = {}
    for alias in _impl._SEQUENCE_ID_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "sequence_id"
            break
    for alias in _impl._UAV_TYPE_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "uav_type"
            break
    frame = frame.rename(columns=rename)
    missing = {"sequence_id", "uav_type"}.difference(frame.columns)
    if missing:
        raise ValueError(f"class-map CSV missing columns: {sorted(missing)}")

    class_map: dict[str, str] = {}
    for _, row in frame.iterrows():
        sequence_id = _class_map_sequence_key(row["sequence_id"])
        uav_type = _class_map_uav_type(row["uav_type"])
        if sequence_id is not None and uav_type is not None:
            class_map[sequence_id] = uav_type
    return class_map


def _canonicalize_sequence_class_map(class_map: dict[Any, Any]) -> dict[str, str]:
    """Drop missing-like sequence IDs from non-CSV class-map formats."""

    normalized: dict[str, str] = {}
    for sequence_id, uav_type in class_map.items():
        normalized_sequence_id = _class_map_sequence_key(sequence_id)
        normalized_uav_type = _class_map_uav_type(uav_type)
        if normalized_sequence_id is not None and normalized_uav_type is not None:
            normalized[normalized_sequence_id] = normalized_uav_type
    return normalized


def _class_map_sequence_key(value: Any) -> str | None:
    try:
        return _impl.parse_official_sequence_cell(value)
    except ValueError:
        return None


def _class_map_uav_type(value: Any) -> str | None:
    if isinstance(value, _impl.np.generic):
        value = value.item()
    return _impl._scalar_to_text(value)


def _read_official_track5_csv(source: Any) -> Any:
    """Read Track 5 CSV input without coercing opaque sequence identifiers."""

    try:
        return _impl.pd.read_csv(source, dtype=str, keep_default_na=False)
    except TypeError:
        return _impl.pd.read_csv(source)


def _load_official_track5_template_file_with_text_sequences(path: Path | str) -> Any:
    """Load requested Track 5 timestamps while preserving zero-padded sequence IDs."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        with _impl.ZipFile(path) as archive:
            if "mmaud_results.csv" not in archive.namelist():
                raise ValueError(f"{path} does not contain 'mmaud_results.csv'")
            with archive.open("mmaud_results.csv") as handle:
                frame = _read_official_track5_csv(_impl.BytesIO(handle.read()))
    else:
        frame = _read_official_track5_csv(path)
    return _impl._normalize_track5_template(frame)


def _read_official_track5_zip_for_validation_with_text_sequences(
    path: Path | str,
    errors: list[str],
) -> tuple[Any | None, list[str], dict[str, Any]]:
    with _impl.ZipFile(path) as archive:
        infos = archive.infolist()
        members = [info.filename for info in infos]
        file_members = [
            _impl._normalized_zip_member_name(info.filename)
            for info in infos
            if not info.is_dir()
        ]
        directory_members = [
            _impl._normalized_zip_member_name(info.filename)
            for info in infos
            if info.is_dir()
        ]
        nested_results = [
            member
            for member in file_members
            if _impl.PurePosixPath(member).name == "mmaud_results.csv"
            and member != "mmaud_results.csv"
        ]
        root_results_count = sum(member == "mmaud_results.csv" for member in file_members)
        root_result_infos = [
            info
            for info in infos
            if not info.is_dir()
            and _impl._normalized_zip_member_name(info.filename) == "mmaud_results.csv"
        ]
        summary = {
            "file_members": file_members,
            "directory_members": directory_members,
            "root_file_members": _impl._root_zip_file_members(file_members),
            "has_root_mmaud_results_csv": root_results_count > 0,
            "root_mmaud_results_csv_count": int(root_results_count),
            "nested_mmaud_results_csv_members": nested_results,
            "contains_only_mmaud_results_csv": (
                file_members == ["mmaud_results.csv"] and not directory_members
            ),
        }
        if not summary["contains_only_mmaud_results_csv"]:
            errors.append("official Track 5 ZIP must contain only mmaud_results.csv")
        if nested_results and root_results_count == 0:
            errors.append(
                "official Track 5 ZIP must place mmaud_results.csv at the archive root"
            )
        if root_results_count == 0:
            return None, members, summary
        if root_results_count > 1:
            errors.append("official Track 5 ZIP contains duplicate mmaud_results.csv members")
        result_info = root_result_infos[0]
        with archive.open(result_info) as handle:
            result_bytes = handle.read()
        summary.update(_impl._official_track5_zip_member_fingerprint(result_info, result_bytes))
        frame = _read_official_track5_csv(_impl.BytesIO(result_bytes))
    return frame, members, summary


def _read_official_track5_results_input_with_text_sequences(
    path: Path | str,
) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return _read_official_track5_csv(path), {
            "input_is_zip": False,
            "source_member": None,
            "source_selection": "csv_file",
        }

    with _impl.ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        root_results = [
            info
            for info in infos
            if _impl._normalized_zip_member_name(info.filename) == "mmaud_results.csv"
        ]
        basename_results = [
            info
            for info in infos
            if _impl.PurePosixPath(
                _impl._normalized_zip_member_name(info.filename)
            ).name
            == "mmaud_results.csv"
        ]
        csv_members = [
            info
            for info in infos
            if _impl.PurePosixPath(
                _impl._normalized_zip_member_name(info.filename)
            ).suffix.lower()
            == ".csv"
        ]
        if len(root_results) > 1:
            raise ValueError("official Track 5 ZIP has duplicate root mmaud_results.csv members")
        if root_results:
            selected = root_results[0]
            selection = "root_mmaud_results_csv"
        elif len(basename_results) == 1:
            selected = basename_results[0]
            selection = "nested_mmaud_results_csv"
        elif len(csv_members) == 1:
            selected = csv_members[0]
            selection = "single_csv_member"
        else:
            raise ValueError(
                "official Track 5 ZIP must contain an unambiguous results CSV: "
                "root mmaud_results.csv, one nested mmaud_results.csv, or one CSV member"
            )
        with archive.open(selected) as handle:
            payload = handle.read()
    return _read_official_track5_csv(_impl.BytesIO(payload)), {
        "input_is_zip": True,
        "source_member": selected.filename,
        "source_selection": selection,
        "input_file_member_count": int(len(infos)),
        "input_csv_member_count": int(len(csv_members)),
    }


def _validate_official_track5_submission_with_text_sequences(
    path: Path | str,
    *,
    template: Any | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
    """Validate Track 5 uploads without stripping leading zeros from Sequence."""

    if timestamp_tolerance_s < 0.0:
        raise ValueError("timestamp_tolerance_s must be non-negative")
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    members: list[str] = []
    zip_summary: dict[str, Any] = {}
    frame: Any | None = None
    is_zip = path.suffix.lower() == ".zip"
    if require_zip and not is_zip:
        errors.append("official Track 5 Codabench upload must be a .zip file")
    try:
        if is_zip:
            frame, members, zip_summary = (
                _read_official_track5_zip_for_validation_with_text_sequences(path, errors)
            )
        else:
            frame = _read_official_track5_csv(path)
    except Exception as exc:
        errors.append(f"could not read official Track 5 submission: {exc}")

    row_diagnostics = _impl.pd.DataFrame()
    schema_summary: dict[str, Any] = {}
    if frame is not None:
        schema_summary, row_diagnostics = _impl._validate_official_track5_frame(
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
        **_impl._official_track5_artifact_fingerprint(path),
        "is_zip": bool(is_zip),
        "require_zip": bool(require_zip),
        "members": members,
        "has_mmaud_results_csv": (
            bool(zip_summary.get("has_root_mmaud_results_csv")) if is_zip else path.exists()
        ),
        "contains_only_mmaud_results_csv": (
            bool(zip_summary.get("contains_only_mmaud_results_csv")) if is_zip else False
        ),
        "expected_columns": list(_impl.OFFICIAL_UG2_RESULT_COLUMNS),
        "timestamp_tolerance_s": float(timestamp_tolerance_s),
        "template_checked": bool(template_checked),
        "errors": errors,
        "warnings": warnings,
        **zip_summary,
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
    blocking_reasons = _impl._official_track5_validation_leaderboard_blocking_reasons(summary)
    summary["score_valid_for_leaderboard"] = bool(not blocking_reasons)
    summary["leaderboard_ready"] = bool(not blocking_reasons)
    summary["codabench_upload_ready"] = bool(
        summary["leaderboard_ready"]
        and summary["is_zip"]
        and summary["contains_only_mmaud_results_csv"]
    )
    summary["leaderboard_blocking_reasons"] = blocking_reasons
    return _impl.OfficialTrack5Validation(summary=summary, rows=row_diagnostics)


def _normalize_official_track5_results_frame_with_domain(frame: Any) -> Any:
    """Normalize Track 5 submission rows with the public class-ID domain check."""

    normalized = _normalize_official_track5_results_frame_original(frame)
    for value in normalized["Classification"]:
        _parse_official_classification_cell_with_domain(value)
    return normalized


def _official_track5_row_diagnostics_with_domain(frame: Any) -> tuple[Any, Any]:
    """Mark out-of-domain Track 5 classifications invalid during validation."""

    diagnostics, normalized = _official_track5_row_diagnostics_original(frame)
    invalid_reasons: dict[int, str] = {}
    if not diagnostics.empty and "Classification" in frame.columns:
        bool_mask = frame["Classification"].astype(str).str.strip().str.lower().isin(
            {"true", "false"}
        )
        if bool_mask.any():
            for row_index in frame.index[bool_mask].astype(int).tolist():
                invalid_reasons[row_index] = (
                    "official MMUAD Classification values must be integer ids, not booleans"
                )
    if not normalized.empty and "classification" in normalized.columns:
        for _, row in normalized.iterrows():
            row_index = int(row["row_index"])
            if row_index in invalid_reasons:
                continue
            try:
                _parse_official_classification_cell_with_domain(row["classification"])
            except ValueError as exc:
                invalid_reasons[row_index] = str(exc)
    if not invalid_reasons:
        return diagnostics, normalized

    diagnostics = diagnostics.copy()
    invalid_indices = set(invalid_reasons)
    mask = diagnostics["row_index"].isin(invalid_indices)
    diagnostics.loc[mask, "status"] = "invalid_classification"
    diagnostics.loc[mask, "reason"] = diagnostics.loc[mask, "row_index"].map(
        invalid_reasons
    )
    if not normalized.empty:
        normalized = normalized.loc[
            ~normalized["row_index"].isin(invalid_indices)
        ].reset_index(drop=True)
    return diagnostics, normalized


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain
_impl.load_sequence_class_map = _load_sequence_class_map_with_official_sequences
_impl.load_official_track5_template_file = (
    _load_official_track5_template_file_with_text_sequences
)
_impl._read_official_track5_zip_for_validation = (
    _read_official_track5_zip_for_validation_with_text_sequences
)
_impl._read_official_track5_results_input = (
    _read_official_track5_results_input_with_text_sequences
)
_impl.normalize_official_track5_results_frame = (
    _normalize_official_track5_results_frame_with_domain
)
_impl._official_track5_row_diagnostics = _official_track5_row_diagnostics_with_domain
_impl.validate_official_track5_submission = (
    _validate_official_track5_submission_with_text_sequences
)

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain
load_sequence_class_map = _load_sequence_class_map_with_official_sequences
load_official_track5_template_file = _load_official_track5_template_file_with_text_sequences
normalize_official_track5_results_frame = (
    _normalize_official_track5_results_frame_with_domain
)
validate_official_track5_submission = (
    _validate_official_track5_submission_with_text_sequences
)
