from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.radar_json import read_radar_table
from raft_uav.mmuad.radar_math import (
    RADAR_AZIMUTH_CONVENTIONS,
    angle_column_to_rad,
    normalize_radar_columns,
    polar_to_cartesian,
    radar_horizontal_std,
)
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_time_column_aliases,
)


def load_radar_polar_csv_as_candidates(
    path: Path,
    *,
    source: str = "radar-polar",
    sequence_id: str | None = None,
    azimuth_convention: str = "north-clockwise",
    angle_unit: str = "deg",
    range_std_m: float = 2.0,
    angle_std_deg: float = 2.0,
    z_std_m: float = 5.0,
) -> CandidateFrame:
    """Load exported polar radar detections and convert them to candidates."""

    return radar_polar_frame_to_candidates(
        read_radar_table(path),
        source=source,
        sequence_id=sequence_id,
        default_sequence_id=Path(path).parent.name,
        azimuth_convention=azimuth_convention,
        angle_unit=angle_unit,
        range_std_m=range_std_m,
        angle_std_deg=angle_std_deg,
        z_std_m=z_std_m,
    )


def radar_polar_frame_to_candidates(
    frame: pd.DataFrame,
    *,
    source: str = "radar-polar",
    sequence_id: str | None = None,
    default_sequence_id: str = "default",
    azimuth_convention: str = "north-clockwise",
    angle_unit: str = "deg",
    range_std_m: float = 2.0,
    angle_std_deg: float = 2.0,
    z_std_m: float = 5.0,
) -> CandidateFrame:
    """Convert an exported polar-radar table frame into candidates.

    Explicit ``*_rad`` columns are radians, explicit ``*_deg`` columns are
    degrees, and generic angle columns continue to use ``angle_unit``.
    """

    normalized = normalize_radar_columns(normalize_time_column_aliases(frame, target="time_s"))
    if sequence_id is not None:
        normalized["sequence_id"] = str(sequence_id)
    elif "sequence_id" not in normalized.columns:
        normalized["sequence_id"] = str(default_sequence_id)
    if "time_s" not in normalized.columns:
        raise ValueError("radar polar table requires time_s/timestamp_s/time column")

    range_m = pd.to_numeric(normalized["range_m"], errors="coerce").to_numpy(float)
    xyz = polar_to_cartesian(
        range_m,
        angle_column_to_rad(normalized, "azimuth", default_angle_unit=angle_unit),
        angle_column_to_rad(
            normalized,
            "elevation",
            default_angle_unit=angle_unit,
            missing_default=0.0,
        ),
        azimuth_convention=azimuth_convention,
    )
    records = pd.DataFrame(
        {
            "sequence_id": normalized["sequence_id"].astype(str),
            "time_s": pd.to_numeric(normalized["time_s"], errors="coerce"),
            "source": str(source),
            "track_id": normalized.get("track_id", np.nan),
            "x_m": xyz[:, 0],
            "y_m": xyz[:, 1],
            "z_m": xyz[:, 2],
            "std_xy_m": radar_horizontal_std(
                range_m,
                angle_std_deg=angle_std_deg,
                range_std_m=range_std_m,
            ),
            "std_z_m": float(z_std_m),
            "confidence": pd.to_numeric(normalized.get("confidence", 1.0), errors="coerce"),
            "class_name": normalized.get("class_name", "uav"),
        }
    )
    return CandidateFrame(
        normalize_candidate_columns(records, default_sequence_id=str(default_sequence_id))
    )
