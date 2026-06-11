"""Small normalized schemas for MMUAD-style UAV tracking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_CANDIDATE_COLUMNS = (
    "sequence_id",
    "time_s",
    "source",
    "track_id",
    "x_m",
    "y_m",
    "z_m",
    "std_xy_m",
    "std_z_m",
    "confidence",
    "class_name",
)

CANONICAL_TRUTH_COLUMNS = (
    "sequence_id",
    "time_s",
    "x_m",
    "y_m",
    "z_m",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "sequence_id": ("sequence", "seq", "scene", "scene_id", "clip", "clip_id"),
    "time_s": ("timestamp_s", "t", "time", "sec", "seconds"),
    "source": ("sensor", "modality"),
    "track_id": ("track", "id", "object_id", "cluster_id", "instance_id"),
    "x_m": ("x", "east_m", "pos_x", "center_x", "cx"),
    "y_m": ("y", "north_m", "pos_y", "center_y", "cy"),
    "z_m": ("z", "up_m", "pos_z", "center_z", "cz"),
    "std_xy_m": ("xy_std_m", "std_m", "position_std_m", "sigma_xy_m"),
    "std_z_m": ("z_std_m", "sigma_z_m"),
    "confidence": ("score", "probability", "cat_prob", "catprob"),
    "class_name": ("class", "label", "category"),
}


@dataclass(frozen=True)
class CandidateFrame:
    """Normalized candidate detections for one or more MMUAD sequences."""

    rows: pd.DataFrame

    def validate(self) -> None:
        missing = {"sequence_id", "time_s", "source", "x_m", "y_m", "z_m"}.difference(
            self.rows.columns
        )
        if missing:
            raise ValueError(f"candidate rows missing required columns: {sorted(missing)}")


@dataclass(frozen=True)
class TruthFrame:
    """Normalized UAV ground-truth positions for one or more MMUAD sequences."""

    rows: pd.DataFrame

    def validate(self) -> None:
        missing = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}.difference(self.rows.columns)
        if missing:
            raise ValueError(f"truth rows missing required columns: {sorted(missing)}")


def normalize_candidate_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
) -> pd.DataFrame:
    """Return a normalized candidate table with canonical column names.

    The official MMUAD archive layout may evolve.  This helper accepts a small
    set of common aliases so exported detector/cluster files can be used
    without rewriting the tracker.
    """

    out = _rename_aliases(frame.copy())
    if out.empty:
        return pd.DataFrame(columns=CANONICAL_CANDIDATE_COLUMNS)
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    if "source" not in out.columns:
        out["source"] = "candidate"
    if "track_id" not in out.columns:
        out["track_id"] = np.nan
    if "std_xy_m" not in out.columns:
        out["std_xy_m"] = 10.0
    if "std_z_m" not in out.columns:
        out["std_z_m"] = out["std_xy_m"]
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    if "class_name" not in out.columns:
        out["class_name"] = "uav"
    missing_required = {"time_s", "x_m", "y_m", "z_m"}.difference(out.columns)
    if missing_required:
        raise ValueError(
            f"candidate table missing required columns: {sorted(missing_required)}; "
            f"available={list(out.columns)}"
        )
    for col in ("time_s", "x_m", "y_m", "z_m", "std_xy_m", "std_z_m", "confidence"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["source"] = out["source"].astype(str)
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s", "source"]).reset_index(drop=True)


def normalize_truth_columns(
    frame: pd.DataFrame,
    *,
    default_sequence_id: str = "default",
) -> pd.DataFrame:
    """Return a normalized truth table with canonical column names."""

    out = _rename_aliases(frame.copy())
    if "sequence_id" not in out.columns:
        out["sequence_id"] = default_sequence_id
    for col in ("time_s", "x_m", "y_m", "z_m"):
        if col not in out.columns:
            raise ValueError(f"truth table missing {col!r}; available={list(out.columns)}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["sequence_id"] = out["sequence_id"].astype(str)
    out = out.loc[np.isfinite(out[["time_s", "x_m", "y_m", "z_m"]]).all(axis=1)].copy()
    return out.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _rename_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {str(col).lower(): col for col in frame.columns}
    rename: dict[Any, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                rename[original] = canonical
                break
    return frame.rename(columns=rename)



def load_jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalar containers into JSON-serializable values."""

    if isinstance(value, dict):
        return {str(key): load_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [load_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return load_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
