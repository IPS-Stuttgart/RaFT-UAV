"""Anchor-aware package wrapper for candidate-mixture multi-start MAP.

The implementation lives in the sibling ``candidate_mixture_map_multistart.py``
file.  This wrapper keeps the public import path while correcting restart
selection when the core smoother uses a non-zero initialization-anchor weight.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.schema import normalize_candidate_columns

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_mixture_map_multistart.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_mixture_map_multistart_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-mixture multi-start implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SELECTION_OBJECTIVE = _IMPL.compute_candidate_mixture_selection_objective

# Export the maintained implementation first; corrected functions below replace
# the two affected callables while preserving all existing public/private helpers.
globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def compute_candidate_mixture_selection_objective(
    result: core.CandidateMixtureMapResult,
    *,
    mixture_config: core.CandidateMixtureMapConfig,
    candidates: pd.DataFrame | None = None,
    initial_estimates: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Evaluate the full restart objective, including its initialization anchor.

    ``candidate_mixture_map`` adds ``anchor_weight * ||x - x_initial||^2`` to the
    trajectory solve.  Because every restart has a different initial trajectory,
    omitting that term makes objective values incomparable whenever
    ``anchor_weight`` is non-zero.
    """

    objective = dict(
        _ORIGINAL_SELECTION_OBJECTIVE(
            result,
            mixture_config=mixture_config,
        )
    )
    anchor_penalty = _trajectory_anchor_penalty(
        candidates=candidates,
        estimates=result.estimates,
        initial_estimates=initial_estimates,
        mixture_config=mixture_config,
    )
    objective["anchor_penalty"] = float(anchor_penalty)
    objective["selection_objective"] = float(
        objective["selection_objective"] + anchor_penalty
    )
    return objective


def run_multistart_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: Any | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> Any:
    """Run every restart and select the lowest full truth-free objective."""

    mixture_config = mixture_config or core.CandidateMixtureMapConfig()
    multistart_config = multistart_config or _IMPL.CandidateMixtureMultiStartConfig()
    starts = _IMPL.build_candidate_mixture_initializations(
        candidates,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external_initial_estimates,
    )
    results: dict[str, core.CandidateMixtureMapResult] = {}
    records: list[dict[str, Any]] = []
    for start_name, initial_estimates in starts.items():
        result = core.run_candidate_mixture_map(
            candidates,
            config=mixture_config,
            initial_estimates=initial_estimates,
            truth=truth,
        )
        objective = compute_candidate_mixture_selection_objective(
            result,
            mixture_config=mixture_config,
            candidates=candidates,
            initial_estimates=initial_estimates,
        )
        pooled = result.summary.get("metrics", {}).get("pooled", {})
        results[start_name] = result
        records.append(
            {
                "start_name": start_name,
                "start_type": start_name.split(":", 1)[0],
                **objective,
                "final_quadratic_surrogate": _IMPL._final_quadratic_surrogate(result),
                "estimate_rows": int(len(result.estimates)),
                "assignment_rows": int(len(result.assignments)),
                "mean_assignment_entropy": _IMPL._column_mean(
                    result.estimates,
                    "mixture_assignment_entropy",
                ),
                "mean_effective_sigma_m": _IMPL._column_mean(
                    result.estimates,
                    "mixture_effective_sigma_m",
                ),
                "diagnostic_mse_3d_m2": _IMPL._optional_float(pooled.get("mse_3d_m2")),
                "diagnostic_rmse_3d_m": _IMPL._optional_float(pooled.get("rmse_3d_m")),
                "diagnostic_p95_3d_m": _IMPL._optional_float(pooled.get("p95_3d_m")),
                "diagnostic_max_3d_m": _IMPL._optional_float(pooled.get("max_3d_m")),
            }
        )

    ranked = pd.DataFrame.from_records(records).sort_values(
        ["selection_objective", "mixture_data_nll", "start_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if ranked.empty:
        raise ValueError("candidate-mixture multi-start produced no starts")
    selected_start = str(ranked.iloc[0]["start_name"])
    ranked["selected"] = ranked["start_name"].astype(str) == selected_start
    summary = {
        "schema": "raft-uav-mmuad-candidate-mixture-multistart-v1",
        "selected_start": selected_start,
        "start_count": int(len(ranked)),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "selection": _IMPL._jsonable(ranked.iloc[0].to_dict()),
        "truth_used_for_selection": False,
    }
    return _IMPL.CandidateMixtureMultiStartResult(
        selected_start=selected_start,
        selected_result=results[selected_start],
        start_summary=ranked,
        initializations=starts,
        summary=_IMPL._jsonable(summary),
    )


def _trajectory_anchor_penalty(
    *,
    candidates: pd.DataFrame | None,
    estimates: pd.DataFrame,
    initial_estimates: pd.DataFrame | None,
    mixture_config: core.CandidateMixtureMapConfig,
) -> float:
    anchor_weight = float(mixture_config.anchor_weight)
    if anchor_weight <= 0.0:
        return 0.0
    if candidates is None:
        raise ValueError(
            "candidates are required to evaluate multi-start selection when anchor_weight > 0"
        )

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    candidate_rows = candidate_rows.reset_index(drop=True)
    candidate_rows["_mixture_input_row"] = np.arange(len(candidate_rows), dtype=int)
    final_rows = pd.DataFrame(estimates).copy()
    if candidate_rows.empty or final_rows.empty:
        return float("inf")
    normalized_initial = core._normalize_initial_estimates(initial_estimates)

    total = 0.0
    for sequence_id, sequence_estimates in final_rows.groupby("sequence_id", sort=True):
        sequence_candidates = candidate_rows.loc[
            candidate_rows["sequence_id"].astype(str) == str(sequence_id)
        ]
        frames = core._prepare_candidate_frames(sequence_candidates, config=mixture_config)
        if not frames:
            return float("inf")
        times = np.asarray([frame["time_s"] for frame in frames], dtype=float)
        anchor_state = core._initial_trajectory(
            frames,
            times=times,
            sequence_id=str(sequence_id),
            initial_estimates=normalized_initial,
            config=mixture_config,
        )
        ordered = sequence_estimates.sort_values("time_s")
        final_times = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(float)
        final_state = ordered[["state_x_m", "state_y_m", "state_z_m"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)
        if (
            len(final_times) != len(times)
            or not np.isfinite(final_times).all()
            or not np.isfinite(final_state).all()
            or not np.allclose(final_times, times, rtol=0.0, atol=1.0e-9)
        ):
            return float("inf")
        total += anchor_weight * float(np.sum((final_state - anchor_state) ** 2))
    return float(total)


# Make the legacy CLI and any function globals resolve the corrected behavior.
_IMPL.compute_candidate_mixture_selection_objective = compute_candidate_mixture_selection_objective
_IMPL.run_multistart_candidate_mixture_map = run_multistart_candidate_mixture_map
globals()["compute_candidate_mixture_selection_objective"] = (
    compute_candidate_mixture_selection_objective
)
globals()["run_multistart_candidate_mixture_map"] = run_multistart_candidate_mixture_map
globals()["_trajectory_anchor_penalty"] = _trajectory_anchor_penalty

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
