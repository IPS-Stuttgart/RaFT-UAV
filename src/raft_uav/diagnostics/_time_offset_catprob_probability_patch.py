"""Enforce probability-domain semantics for time-offset radar class thresholds."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Any


_time_offset = import_module("raft_uav.diagnostics.time_offset")
_PATCH_MARKER = "_raft_uav_time_offset_catprob_probability_patch_applied"
_ORIGINAL_CATPROB_CANDIDATE_POOL = _time_offset.catprob_candidate_pool
_ORIGINAL_RUN_TIME_OFFSET_DIAGNOSTIC = _time_offset.run_time_offset_diagnostic


def _probability_control(value: object, *, name: str) -> float:
    """Return a finite probability in the closed interval [0, 1]."""

    number = _time_offset._finite_real_control(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


@wraps(_ORIGINAL_CATPROB_CANDIDATE_POOL)
def catprob_candidate_pool(candidates: Any, threshold: object) -> Any:
    """Apply catProb filtering only with a valid probability threshold."""

    if threshold is None:
        return _ORIGINAL_CATPROB_CANDIDATE_POOL(candidates, threshold)
    return _ORIGINAL_CATPROB_CANDIDATE_POOL(
        candidates,
        _probability_control(threshold, name="threshold"),
    )


@wraps(_ORIGINAL_RUN_TIME_OFFSET_DIAGNOSTIC)
def run_time_offset_diagnostic(
    *,
    dataset_root: Path,
    flight_name: str,
    source: str,
    tau_min_s: float,
    tau_max_s: float,
    tau_step_s: float,
    dimensions: str = "auto",
    radar_selection: str = "oracle-nearest-truth",
    radar_catprob_threshold: float | None = 0.4,
    max_truth_time_delta_s: float = 2.0,
    objective: str = "p95",
    output_dir: Path = Path("outputs/time-offset"),
    write_plot: bool = True,
) -> dict[str, Any]:
    """Reject out-of-domain radar class thresholds before accessing dataset files."""

    normalized_threshold = radar_catprob_threshold
    if source == "radar" and radar_catprob_threshold is not None:
        normalized_threshold = _probability_control(
            radar_catprob_threshold,
            name="radar_catprob_threshold",
        )
    return _ORIGINAL_RUN_TIME_OFFSET_DIAGNOSTIC(
        dataset_root=dataset_root,
        flight_name=flight_name,
        source=source,
        tau_min_s=tau_min_s,
        tau_max_s=tau_max_s,
        tau_step_s=tau_step_s,
        dimensions=dimensions,
        radar_selection=radar_selection,
        radar_catprob_threshold=normalized_threshold,
        max_truth_time_delta_s=max_truth_time_delta_s,
        objective=objective,
        output_dir=output_dir,
        write_plot=write_plot,
    )


def install() -> None:
    """Install probability-domain validation once per interpreter."""

    if getattr(_time_offset, _PATCH_MARKER, False):
        return
    _time_offset.catprob_candidate_pool = catprob_candidate_pool
    _time_offset.run_time_offset_diagnostic = run_time_offset_diagnostic
    legacy = getattr(_time_offset, "_legacy", None)
    if legacy is not None:
        legacy.catprob_candidate_pool = catprob_candidate_pool
        legacy.run_time_offset_diagnostic = run_time_offset_diagnostic
    setattr(_time_offset, _PATCH_MARKER, True)


install()
