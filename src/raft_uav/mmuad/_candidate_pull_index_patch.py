"""Keep candidate-pull result labels separate from positional coordinates."""

from __future__ import annotations

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


def install() -> None:
    """Install positional candidate-center indexing exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_pull as candidate_pull

    original = candidate_pull.candidate_centers_for_results

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

        try:
            row_positions = pd.to_numeric(
                centers["row_index"],
                errors="raise",
            ).to_numpy(dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("candidate-pull returned invalid row positions") from exc
        if np.any((row_positions < 0) | (row_positions >= len(original_index))):
            raise RuntimeError("candidate-pull returned out-of-range row positions")

        out = centers.copy()
        out["row_index"] = original_index[row_positions]
        return out

    candidate_pull.candidate_centers_for_results = candidate_centers_for_results
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation.candidate_centers_for_results = candidate_centers_for_results
    _INSTALLED = True
