"""Lightweight UAV type helpers for MMUAD/UG2-style submissions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame


UNKNOWN_LABELS = {"", "unknown", "nan", "none", "uav", "drone"}


def infer_sequence_class_map_from_candidates(
    candidates: CandidateFrame,
    *,
    min_confidence: float = 0.0,
    default_class: str = "unknown",
) -> dict[str, str]:
    """Infer one UAV type per sequence from weighted candidate class votes."""

    rows = candidates.rows.copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return {}
    sequence_ids = sorted(str(sequence_id) for sequence_id in rows["sequence_id"].dropna().unique())
    if "class_name" not in rows.columns:
        return {sequence_id: str(default_class) for sequence_id in sequence_ids}
    rows["confidence"] = pd.to_numeric(rows.get("confidence", 1.0), errors="coerce").fillna(1.0)
    rows = rows.loc[rows["confidence"] >= float(min_confidence)].copy()
    result: dict[str, str] = {sequence_id: str(default_class) for sequence_id in sequence_ids}
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        votes: dict[str, float] = {}
        for _, row in group.iterrows():
            label = str(row.get("class_name", default_class)).strip()
            if label.lower() in UNKNOWN_LABELS:
                continue
            votes[label] = votes.get(label, 0.0) + float(row.get("confidence", 1.0))
        if votes:
            result[str(sequence_id)] = sorted(
                votes.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0]
    return result


def class_map_to_frame(class_map: dict[str, str]) -> pd.DataFrame:
    """Return a stable two-column class-map table."""

    return pd.DataFrame(
        {"sequence_id": list(class_map.keys()), "uav_type": list(class_map.values())}
    ).sort_values("sequence_id").reset_index(drop=True)


def write_sequence_class_map(class_map: dict[str, str], path: Path) -> Path:
    """Write a sequence class-map CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    class_map_to_frame(class_map).to_csv(path, index=False)
    return path
