"""Compatibility guards for guarded-tournament output publication.

The maintained implementation lives in the sibling ``tournament.py`` module.
This package preserves the public import path while preventing selected-output
refreshes from deleting their own candidate input and ensuring
``require_improvement`` failures occur before raw-fallback artifacts are
published.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from pathlib import Path
import sys
from threading import RLock
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tournament.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._tournament_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load tournament implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_GUARDED_TOURNAMENT = _IMPL.run_guarded_tournament
_ORIGINAL_COPY_SELECTED_PREDICTIONS = _IMPL._copy_selected_predictions
_RUN_GUARD = RLock()


def _selected_prediction_copy_target(path: Path, output_dir: Path) -> Path:
    """Return the path replaced by the tournament's selected-output refresh."""

    if path.is_dir():
        return output_dir / "selected_predictions"
    suffix = "".join(path.suffixes) or ".bin"
    return output_dir / f"selected_predictions{suffix}"


def _copy_selected_predictions(path: Path, output_dir: Path) -> None:
    """Copy a selected candidate without deleting that candidate first."""

    source = Path(path)
    destination_root = Path(output_dir)
    target = _selected_prediction_copy_target(source, destination_root)
    source_resolved = source.resolve()
    target_resolved = target.resolve()
    if source_resolved == target_resolved or source_resolved.is_relative_to(
        target_resolved
    ):
        raise ValueError(
            "selected candidate path must not be the selected_predictions copy "
            "target or live inside it"
        )
    _ORIGINAL_COPY_SELECTED_PREDICTIONS(source, destination_root)


def _remove_new_empty_output_dir(output_dir: Path, *, existed: bool) -> None:
    """Undo the legacy eager mkdir when a guarded run intentionally fails."""

    if existed:
        return
    try:
        output_dir.rmdir()
    except OSError:
        pass


def run_guarded_tournament(
    raw_prediction_path: Path,
    truth_dir: Path,
    output_dir: Path,
    *,
    candidates: Sequence[tuple[str, Path]] = (),
    fold_count: int = 5,
    seed: int = 0,
    expected_sequence_count: int = 102,
    sequences: Sequence[str] = (),
    bootstrap_samples: int = 5000,
    min_mean_hota_gain: float = 0.001,
    min_ci_hota_gain: float = 0.0,
    max_mean_mota_drop: float = 0.002,
    max_mean_idf1_drop: float = 0.002,
    max_worst_scenario_hota_drop: float = 0.01,
    require_complete: bool = True,
    require_improvement: bool = False,
    copy_selected: bool = True,
) -> Any:
    """Run the tournament without publishing a forbidden raw fallback."""

    with _RUN_GUARD:
        if not require_improvement:
            return _ORIGINAL_RUN_GUARDED_TOURNAMENT(
                raw_prediction_path,
                truth_dir,
                output_dir,
                candidates=candidates,
                fold_count=fold_count,
                seed=seed,
                expected_sequence_count=expected_sequence_count,
                sequences=sequences,
                bootstrap_samples=bootstrap_samples,
                min_mean_hota_gain=min_mean_hota_gain,
                min_ci_hota_gain=min_ci_hota_gain,
                max_mean_mota_drop=max_mean_mota_drop,
                max_mean_idf1_drop=max_mean_idf1_drop,
                max_worst_scenario_hota_drop=max_worst_scenario_hota_drop,
                require_complete=require_complete,
                require_improvement=False,
                copy_selected=copy_selected,
            )

        output_path = Path(output_dir)
        output_existed = output_path.exists()
        deferred_copies: list[tuple[Path, Path]] = []
        deferred_writes: list[tuple[Any, dict[str, Any]]] = []

        def defer_copy(path: Path, destination: Path) -> None:
            deferred_copies.append((Path(path), Path(destination)))

        def defer_write(result: Any, **write_kwargs: Any) -> None:
            deferred_writes.append((result, write_kwargs))

        previous_copy = _IMPL._copy_selected_predictions
        previous_write = _IMPL._write_outputs
        _IMPL._copy_selected_predictions = defer_copy
        _IMPL._write_outputs = defer_write
        try:
            result = _ORIGINAL_RUN_GUARDED_TOURNAMENT(
                raw_prediction_path,
                truth_dir,
                output_path,
                candidates=candidates,
                fold_count=fold_count,
                seed=seed,
                expected_sequence_count=expected_sequence_count,
                sequences=sequences,
                bootstrap_samples=bootstrap_samples,
                min_mean_hota_gain=min_mean_hota_gain,
                min_ci_hota_gain=min_ci_hota_gain,
                max_mean_mota_drop=max_mean_mota_drop,
                max_mean_idf1_drop=max_mean_idf1_drop,
                max_worst_scenario_hota_drop=max_worst_scenario_hota_drop,
                require_complete=require_complete,
                require_improvement=False,
                copy_selected=copy_selected,
            )
        finally:
            _IMPL._copy_selected_predictions = previous_copy
            _IMPL._write_outputs = previous_write

        selected = next(row for row in result.rows if row.selected)
        if selected.is_raw:
            _remove_new_empty_output_dir(output_path, existed=output_existed)
            raise RuntimeError(
                "no transformed candidate cleared the configured guards; "
                "raw fallback selected"
            )

        for source, destination in deferred_copies:
            previous_copy(source, destination)
        for write_result, write_kwargs in deferred_writes:
            previous_write(write_result, **write_kwargs)
        return result


_IMPL._selected_prediction_copy_target = _selected_prediction_copy_target
_IMPL._copy_selected_predictions = _copy_selected_predictions
_IMPL._remove_new_empty_output_dir = _remove_new_empty_output_dir
_IMPL.run_guarded_tournament = run_guarded_tournament

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_selected_prediction_copy_target"] = _selected_prediction_copy_target
globals()["_copy_selected_predictions"] = _copy_selected_predictions
globals()["_remove_new_empty_output_dir"] = _remove_new_empty_output_dir
globals()["run_guarded_tournament"] = run_guarded_tournament

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
