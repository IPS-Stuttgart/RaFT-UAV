"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from raft_uav.mmuad import _submission_impl as _impl


_parse_original = _impl.parse_official_classification_cell


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        raise ValueError(f"invalid official Track 5 class id: {class_id!r}")
    return class_id


def _read_official_track5_csv(source: Any) -> Any:
    """Read an official Track 5 CSV without coercing sequence IDs.

    Pandas' default dtype inference converts numeric-looking sequence identifiers
    such as ``001`` to integers.  The official Track 5 schema treats ``Sequence``
    as an opaque identifier, so all official CSV/ZIP readers keep cells as text
    and let the existing validators parse numeric timestamp/classification fields.
    """

    return _impl.pd.read_csv(source, dtype=str, keep_default_na=False)


def _load_official_track5_template_file_preserving_sequences(path: Path) -> Any:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            if "mmaud_results.csv" not in archive.namelist():
                raise ValueError(f"{path} does not contain 'mmaud_results.csv'")
            with archive.open("mmaud_results.csv") as handle:
                frame = _read_official_track5_csv(BytesIO(handle.read()))
    else:
        frame = _read_official_track5_csv(path)
    return _impl._normalize_track5_template(frame)


def _read_official_track5_zip_for_validation_preserving_sequences(
    path: Path,
    errors: list[str],
) -> tuple[Any | None, list[str], dict[str, Any]]:
    with ZipFile(path) as archive:
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
            if PurePosixPath(member).name == "mmaud_results.csv"
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
        frame = _read_official_track5_csv(BytesIO(result_bytes))
    return frame, members, summary


def _read_official_track5_results_input_preserving_sequences(
    path: Path,
) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return _read_official_track5_csv(path), {
            "input_is_zip": False,
            "source_member": None,
            "source_selection": "csv_file",
        }

    with ZipFile(path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        root_results = [
            info
            for info in infos
            if _impl._normalized_zip_member_name(info.filename) == "mmaud_results.csv"
        ]
        basename_results = [
            info
            for info in infos
            if PurePosixPath(_impl._normalized_zip_member_name(info.filename)).name
            == "mmaud_results.csv"
        ]
        csv_members = [
            info
            for info in infos
            if PurePosixPath(_impl._normalized_zip_member_name(info.filename)).suffix.lower()
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
    return _read_official_track5_csv(BytesIO(payload)), {
        "input_is_zip": True,
        "source_member": selected.filename,
        "source_selection": selection,
        "input_file_member_count": int(len(infos)),
        "input_csv_member_count": int(len(csv_members)),
    }


def _validate_official_track5_submission_preserving_sequences(
    path: Path,
    *,
    template: Any | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
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
            frame, members, zip_summary = _impl._read_official_track5_zip_for_validation(
                path,
                errors,
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


def _inspect_submission_zip_preserving_sequences(path: Path) -> dict[str, Any]:
    path = Path(path)
    with ZipFile(path) as archive:
        names = archive.namelist()
        has_mmaud = "mmaud_results.csv" in names
        row_count = None
        columns: list[str] | None = None
        if has_mmaud:
            with archive.open("mmaud_results.csv") as handle:
                frame = _read_official_track5_csv(BytesIO(handle.read()))
            row_count = int(len(frame))
            columns = list(frame.columns)
    return {
        "path": str(path),
        "members": names,
        "has_mmaud_results_csv": has_mmaud,
        "row_count": row_count,
        "columns": columns,
    }


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain
_impl.load_official_track5_template_file = _load_official_track5_template_file_preserving_sequences
_impl._read_official_track5_zip_for_validation = (  # type: ignore[attr-defined]
    _read_official_track5_zip_for_validation_preserving_sequences
)
_impl._read_official_track5_results_input = (  # type: ignore[attr-defined]
    _read_official_track5_results_input_preserving_sequences
)
_impl.validate_official_track5_submission = _validate_official_track5_submission_preserving_sequences
_impl.inspect_submission_zip = _inspect_submission_zip_preserving_sequences

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain
