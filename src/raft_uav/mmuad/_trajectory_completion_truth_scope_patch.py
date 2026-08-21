"""Keep trajectory-completion truth diagnostics inside physical flight scopes."""

from __future__ import annotations

import pandas as pd


def _truth_for_estimate_group(
    truth_rows: pd.DataFrame,
    group: pd.DataFrame,
) -> pd.DataFrame:
    """Return truth rows matching one estimate group's sequence and flight."""

    sequence_id = str(group["sequence_id"].iloc[0])
    scoped = truth_rows.loc[
        truth_rows["sequence_id"].astype(str) == sequence_id
    ].copy()
    if (
        scoped.empty
        or "flight_id" not in group.columns
        or "flight_id" not in scoped.columns
    ):
        return scoped

    flight_values = group["flight_id"].drop_duplicates()
    if len(flight_values) != 1:
        return scoped.iloc[0:0].copy()
    flight_id = flight_values.iloc[0]
    if pd.isna(flight_id):
        return scoped.loc[scoped["flight_id"].isna()].copy()
    return scoped.loc[
        scoped["flight_id"].notna()
        & scoped["flight_id"].astype(str).eq(str(flight_id))
    ].copy()


def _attach_truth_errors_by_physical_flight(
    estimates: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach truth errors without mixing reused sequence IDs across flights."""

    if estimates.empty or truth_rows is None or truth_rows.empty:
        return estimates.copy()

    group_columns = ["sequence_id"]
    flight_scoped = (
        "flight_id" in estimates.columns and "flight_id" in truth_rows.columns
    )
    if flight_scoped:
        group_columns.append("flight_id")

    from raft_uav.mmuad.tracker import add_truth_errors

    frames: list[pd.DataFrame] = []
    for _, group in estimates.groupby(group_columns, sort=True, dropna=False):
        scoped_truth = _truth_for_estimate_group(truth_rows, group)
        if scoped_truth.empty and not flight_scoped:
            frames.append(group.copy())
        else:
            frames.append(add_truth_errors(group.copy(), scoped_truth))
    return pd.concat(frames, ignore_index=True) if frames else estimates.copy()


def install() -> None:
    """Install flight-aware truth attachment in the public and legacy modules."""

    from raft_uav.mmuad import trajectory_completion

    trajectory_completion._attach_truth_errors_by_sequence = (
        _attach_truth_errors_by_physical_flight
    )
    implementation = getattr(trajectory_completion, "_IMPL", None)
    if implementation is not None:
        implementation._attach_truth_errors_by_sequence = (
            _attach_truth_errors_by_physical_flight
        )
