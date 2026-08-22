"""Keep candidate-pull row indexing safe and artifact outputs distinct."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_INSTALLED = False


def _contains_complex_values(values: np.ma.MaskedArray) -> bool:
    """Return whether visible values contain any complex scalar."""

    if np.iscomplexobj(values):
        return True
    if values.dtype != object:
        return False
    return any(
        np.iscomplexobj(np.asanyarray(item))
        for item in values.compressed().reshape(-1)
    )


def _current_positions(current_xyz: object, *, row_count: int) -> np.ndarray:
    """Return one finite real position row per result row."""

    shape_message = "current_xyz must have shape (len(results), 3)"
    value_message = "current_xyz must contain only finite real values"
    try:
        masked_positions = np.ma.asarray(current_xyz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(shape_message) from exc
    if _contains_complex_values(masked_positions):
        raise ValueError(value_message)
    if bool(np.ma.getmaskarray(masked_positions).any()):
        raise ValueError(value_message)
    try:
        positions = np.asarray(masked_positions.filled(np.nan), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(shape_message) from exc
    if positions.ndim != 2 or positions.shape != (row_count, 3):
        raise ValueError(shape_message)
    if not bool(np.isfinite(positions).all()):
        raise ValueError(value_message)
    return positions


def _candidate_row_positions(row_index: pd.Series) -> np.ndarray:
    """Return candidate row positions without lossy numeric coercion."""

    message = "candidate-pull returned invalid row positions"
    try:
        numeric_positions = pd.to_numeric(row_index, errors="raise")
        validated: list[int] = []
        for value in numeric_positions.to_numpy():
            if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
                raise ValueError(message)
            if isinstance(value, (int, np.integer)):
                position = int(value)
            else:
                numeric_value = float(value)
                if not np.isfinite(numeric_value) or not numeric_value.is_integer():
                    raise ValueError(message)
                position = int(numeric_value)
            if position < np.iinfo(np.int64).min or position > np.iinfo(np.int64).max:
                raise ValueError(message)
            validated.append(position)
        return np.asarray(validated, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(message) from exc


def _canonical_output_path(path: Path) -> str:
    """Return a normalized path key without requiring the output to exist."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        resolved = candidate.absolute()
    return os.path.normcase(os.fspath(resolved))


def _paths_alias(left: Path, right: Path) -> bool:
    """Return whether two output paths address the same filesystem object."""

    if _canonical_output_path(left) == _canonical_output_path(right):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _validate_artifact_output_paths(
    *,
    results_csv: Path,
    submission_zip: Path | None,
    provenance_json: Path | None,
    centers_csv: Path | None,
    sequence_features_csv: Path | None,
    alpha_assignments_csv: Path | None,
) -> None:
    """Reject colliding candidate-pull outputs before any path is mutated."""

    outputs = [
        ("results_csv", Path(results_csv)),
        ("submission_zip", None if submission_zip is None else Path(submission_zip)),
        ("provenance_json", None if provenance_json is None else Path(provenance_json)),
        ("centers_csv", None if centers_csv is None else Path(centers_csv)),
        (
            "sequence_features_csv",
            None if sequence_features_csv is None else Path(sequence_features_csv),
        ),
        (
            "alpha_assignments_csv",
            None if alpha_assignments_csv is None else Path(alpha_assignments_csv),
        ),
    ]
    present = [(name, path) for name, path in outputs if path is not None]
    for index, (left_name, left_path) in enumerate(present):
        for right_name, right_path in present[index + 1 :]:
            if not _paths_alias(left_path, right_path):
                continue
            resolved = _canonical_output_path(left_path)
            raise ValueError(
                "candidate-pull output paths must be distinct: "
                f"{left_name} and {right_name} both refer to {resolved}"
            )


def install() -> None:
    """Install candidate-pull safety guards exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_pull as candidate_pull

    original = candidate_pull.candidate_centers_for_results
    original_write_artifacts = candidate_pull.write_candidate_pull_artifacts

    def candidate_centers_for_results(
        candidates: pd.DataFrame,
        results: pd.DataFrame,
        current_xyz: np.ndarray,
        *,
        top_k: int = 5,
        time_tolerance_s: float = 0.5,
    ) -> pd.DataFrame:
        """Build centers positionally while preserving unique result labels."""

        result_rows = pd.DataFrame(results).copy()
        if not result_rows.index.is_unique:
            raise ValueError("results index must be unique")
        positions = _current_positions(current_xyz, row_count=len(result_rows))
        original_index = result_rows.index.to_numpy(copy=True)
        normalized_results = result_rows.reset_index(drop=True)
        centers = original(
            candidates,
            normalized_results,
            positions,
            top_k=top_k,
            time_tolerance_s=time_tolerance_s,
        )
        if centers.empty or "row_index" not in centers.columns:
            return centers

        row_positions = _candidate_row_positions(centers["row_index"])
        if np.any((row_positions < 0) | (row_positions >= len(original_index))):
            raise RuntimeError("candidate-pull returned out-of-range row positions")

        out = centers.copy()
        out["row_index"] = original_index[row_positions]
        return out

    def write_candidate_pull_artifacts(
        result: Any,
        *,
        results_csv: Path,
        submission_zip: Path | None = None,
        provenance_json: Path | None = None,
        centers_csv: Path | None = None,
        sequence_features_csv: Path | None = None,
        alpha_assignments_csv: Path | None = None,
    ) -> dict[str, str]:
        """Write candidate-pull artifacts only to mutually distinct paths."""

        _validate_artifact_output_paths(
            results_csv=results_csv,
            submission_zip=submission_zip,
            provenance_json=provenance_json,
            centers_csv=centers_csv,
            sequence_features_csv=sequence_features_csv,
            alpha_assignments_csv=alpha_assignments_csv,
        )
        return original_write_artifacts(
            result,
            results_csv=results_csv,
            submission_zip=submission_zip,
            provenance_json=provenance_json,
            centers_csv=centers_csv,
            sequence_features_csv=sequence_features_csv,
            alpha_assignments_csv=alpha_assignments_csv,
        )

    candidate_pull.candidate_centers_for_results = candidate_centers_for_results
    candidate_pull.write_candidate_pull_artifacts = write_candidate_pull_artifacts
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation.candidate_centers_for_results = candidate_centers_for_results
        implementation.write_candidate_pull_artifacts = write_candidate_pull_artifacts
    _INSTALLED = True
