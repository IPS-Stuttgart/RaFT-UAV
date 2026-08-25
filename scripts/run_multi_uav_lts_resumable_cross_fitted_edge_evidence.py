#!/usr/bin/env python3
"""Run cross-fitted Multi-UAV LTS evidence with persistent stage reuse."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import run_multi_uav_lts_cross_fitted_edge_evidence as crossfit

_SCHEMA = "raft-uav-multi-uav-lts-resumable-cross-fitted-evidence-v1"
_COMPACT_FILE_LIMIT_BYTES = 32 * 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _argument_path(arguments: Sequence[str], flag: str) -> Path:
    values = list(arguments)
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise ValueError(f"missing required argument {flag}") from exc
    if index + 1 >= len(values):
        raise ValueError(f"missing value for {flag}")
    return Path(values[index + 1]).expanduser().resolve()


def _replace_argument(arguments: Sequence[str], flag: str, value: Path) -> list[str]:
    result = list(arguments)
    try:
        index = result.index(flag)
    except ValueError as exc:
        raise ValueError(f"missing required argument {flag}") from exc
    if index + 1 >= len(result):
        raise ValueError(f"missing value for {flag}")
    result[index + 1] = str(value)
    return result


def _persistent_run_dir(requested_run_dir: Path) -> Path:
    work_root = os.environ.get("WORK_ROOT", "").strip()
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if not work_root or not github_sha:
        return requested_run_dir
    safe_sha = "".join(character for character in github_sha if character.isalnum())
    if not safe_sha:
        raise ValueError("GITHUB_SHA does not contain a usable cache key")
    return (
        Path(work_root).expanduser().resolve()
        / "cross-fitted-edge-resume"
        / safe_sha
    )


def _complete_candidate(
    root: Path,
    *,
    summary_name: str,
    candidate_name: str,
    expected_sequences: int,
) -> Path | None:
    output_dir = root / "predictions"
    summary_path = root / summary_name
    if not summary_path.is_file() or not output_dir.is_dir():
        return None
    try:
        summary = _load_json(summary_path)
        digest, total_bytes, count = crossfit.evidence._directory_digest(output_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        summary.get("candidate") != candidate_name
        or int(summary.get("sequence_count", -1)) != expected_sequences
        or count != expected_sequences
        or total_bytes <= 0
        or summary.get("prediction_content_sha256") != digest
        or int(summary.get("prediction_content_bytes", -1)) != total_bytes
    ):
        return None
    return output_dir


def _scorecard_is_fresh(metrics_path: Path, prediction_dir: Path) -> bool:
    try:
        prediction_times = [
            path.stat().st_mtime_ns for path in prediction_dir.glob("*.txt")
        ]
        return bool(prediction_times) and metrics_path.stat().st_mtime_ns >= max(
            prediction_times
        )
    except OSError:
        return False


def _fit_models_resumable(
    original: Callable[..., Mapping[int, Any]],
    proposal_dir: Path,
    truth_dir: Path,
    folds: tuple[Any, ...],
    *,
    run_dir: Path,
) -> dict[int, Any]:
    models: dict[int, Any] = {}
    missing: list[Any] = []
    for fold in folds:
        root = run_dir / "cross-fitted-edge-models" / f"fold-{fold.index}"
        model_path = root / "edge-model.json"
        summary_path = root / "fit-summary.json"
        try:
            models[fold.index] = crossfit._validate_model_provenance(
                model_path,
                summary_path,
                fold,
            )
            print(f"Reusing edge model for fold {fold.index}", flush=True)
        except (OSError, ValueError, json.JSONDecodeError):
            missing.append(fold)
    if missing:
        models.update(
            original(
                proposal_dir,
                truth_dir,
                tuple(missing),
                run_dir=run_dir,
            )
        )
    return models


def _run_static_resumable(
    original: Callable[..., Mapping[str, Path]],
    proposal_dir: Path,
    seed_dir: Path,
    *,
    image_root: Path,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    candidates = crossfit.evidence.CANDIDATES
    outputs: dict[str, Path] = {}
    missing: list[tuple[str, tuple[str, ...]]] = []
    for name, arguments in candidates:
        complete = _complete_candidate(
            run_dir / name,
            summary_name="native-resolution-summary.json",
            candidate_name=name,
            expected_sequences=expected_sequences,
        )
        if complete is None:
            missing.append((name, arguments))
        else:
            print(f"Reusing complete candidate {name}", flush=True)
            outputs[name] = complete
    if not missing:
        return outputs

    crossfit.evidence.CANDIDATES = tuple(missing)
    try:
        generated = original(
            proposal_dir,
            seed_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
    finally:
        crossfit.evidence.CANDIDATES = candidates
    overlap = set(outputs) & set(generated)
    if overlap:
        raise ValueError(f"duplicate resumed static candidates: {sorted(overlap)}")
    return {**outputs, **generated}


def _materialize_oof_resumable(
    original: Callable[..., Path],
    *args: Any,
    **kwargs: Any,
) -> Path:
    name = str(args[0])
    run_dir = Path(kwargs["run_dir"])
    expected_sequences = int(kwargs["expected_sequences"])
    complete = _complete_candidate(
        run_dir / name,
        summary_name="cross-fitted-candidate-summary.json",
        candidate_name=name,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete candidate {name}", flush=True)
        return complete
    return original(*args, **kwargs)


def _score_resumable(
    original: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    name = str(args[0])
    prediction_dir = Path(args[1])
    run_dir = Path(kwargs["run_dir"])
    expected_sequences = int(kwargs["expected_sequences"])
    metrics_path = run_dir / name / "metrics.json"
    if metrics_path.is_file() and _scorecard_is_fresh(metrics_path, prediction_dir):
        try:
            payload = _load_json(metrics_path)
            if int(payload.get("sequence_count", -1)) == expected_sequences:
                print(f"Reusing scorecard for {name}", flush=True)
                return payload
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return original(*args, **kwargs)


def _clear_terminal_state(run_dir: Path) -> None:
    for path in (
        run_dir / "evidence-summary.json",
        run_dir / "evidence-failure.json",
        run_dir / "tournament" / "tournament_summary.json",
    ):
        path.unlink(missing_ok=True)


def _write_plan(run_dir: Path, requested_run_dir: Path) -> None:
    crossfit.evidence._write_json(
        run_dir / "resumable-plan.json",
        {
            "schema": _SCHEMA,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "requested_run_dir": str(requested_run_dir),
            "persistent_run_dir": str(run_dir),
            "resume_enabled": run_dir != requested_run_dir,
            "reused_units": [
                "validated fold models",
                "complete static candidates",
                "complete out-of-fold candidates",
                "fresh scorecards",
            ],
        },
    )


def _sync_compact_evidence(source: Path, destination: Path) -> None:
    if source == destination or not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if "predictions" in relative.parts:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _COMPACT_FILE_LIMIT_BYTES:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def main() -> int:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return crossfit.main()

    requested_run_dir = _argument_path(sys.argv[1:], "--run-dir")
    persistent_run_dir = _persistent_run_dir(requested_run_dir)
    requested_run_dir.mkdir(parents=True, exist_ok=True)
    persistent_run_dir.mkdir(parents=True, exist_ok=True)
    original_argv = list(sys.argv)
    sys.argv = [
        sys.argv[0],
        *_replace_argument(sys.argv[1:], "--run-dir", persistent_run_dir),
    ]
    _clear_terminal_state(persistent_run_dir)
    _write_plan(persistent_run_dir, requested_run_dir)

    original_static_runner = crossfit.improved._run_candidates_with_native_dimensions
    original_fit = crossfit._fit_fold_models
    original_oof_candidate = crossfit._materialize_out_of_fold_candidate
    original_score = crossfit.evidence._score

    def static_runner(*args: Any, **kwargs: Any) -> dict[str, Path]:
        return _run_static_resumable(original_static_runner, *args, **kwargs)

    def fit_models(*args: Any, **kwargs: Any) -> dict[int, Any]:
        return _fit_models_resumable(original_fit, *args, **kwargs)

    def materialize_oof(*args: Any, **kwargs: Any) -> Path:
        return _materialize_oof_resumable(
            original_oof_candidate,
            *args,
            **kwargs,
        )

    def score(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _score_resumable(original_score, *args, **kwargs)

    crossfit.improved._run_candidates_with_native_dimensions = static_runner
    crossfit._fit_fold_models = fit_models
    crossfit._materialize_out_of_fold_candidate = materialize_oof
    crossfit.evidence._score = score
    try:
        return crossfit.main()
    finally:
        crossfit.improved._run_candidates_with_native_dimensions = original_static_runner
        crossfit._fit_fold_models = original_fit
        crossfit._materialize_out_of_fold_candidate = original_oof_candidate
        crossfit.evidence._score = original_score
        sys.argv = original_argv
        _sync_compact_evidence(persistent_run_dir, requested_run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
