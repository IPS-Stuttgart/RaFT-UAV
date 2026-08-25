#!/usr/bin/env python3
"""Run resumable metric-aware Multi-UAV LTS evidence.

The legacy proposal-graph ladder already has its own guarded workflow.  This runner
therefore evaluates the metric-aware/Bayesian ladder against the raw control by
default, while retaining opt-in ``best`` and ``all`` legacy scopes.  Long-lived
outputs are cached by commit so interrupted self-hosted runs can reuse completed
fold models, candidates, and scorecards.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import run_multi_uav_lts_metric_aware_evidence as metric
from raft_uav.multi_uav_lts.scenario_box_policy import (
    assemble_scenario_policy_predictions,
    write_summary,
)

_POLICY_NAME = "graph_metric_edge_swarm_beam_oof_scenario_policy"
_REFERENCE_PREFIX = "scenario-policy-reference"
_GAP_VARIANTS = (
    (f"{metric._METRIC_GRAPH_NAME}_rts_gap1", 1),
    (f"{metric._METRIC_GRAPH_NAME}_rts_gap2", 2),
)
_LEGACY_SCOPES = frozenset({"none", "best", "all"})
_RESUME_SCHEMA = "raft-uav-multi-uav-lts-resumable-metric-evidence-v1"
_COMPACT_FILE_LIMIT_BYTES = 32 * 1024 * 1024


def _variant_key(candidate_name: str) -> str:
    prefix = metric._METRIC_GRAPH_NAME + "_"
    return candidate_name.removeprefix(prefix)


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


def _legacy_scope() -> str:
    value = os.environ.get("RAFT_UAV_LTS_LEGACY_SCOPE", "none").strip().lower()
    if value not in _LEGACY_SCOPES:
        choices = ", ".join(sorted(_LEGACY_SCOPES))
        raise ValueError(
            "RAFT_UAV_LTS_LEGACY_SCOPE must be one of "
            f"{choices}; received {value!r}"
        )
    return value


def _persistent_run_dir(requested_run_dir: Path, scope: str) -> Path:
    work_root = os.environ.get("WORK_ROOT", "").strip()
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if not work_root or not github_sha:
        return requested_run_dir
    safe_sha = "".join(character for character in github_sha if character.isalnum())
    if not safe_sha:
        raise ValueError("GITHUB_SHA does not contain a usable cache key")
    return (
        Path(work_root).expanduser().resolve()
        / "metric-aware-resume"
        / safe_sha
        / scope
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
        digest, total_bytes, count = metric.crossfit.evidence._directory_digest(
            output_dir
        )
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
    validator: Callable[[Path, Path, Any], Any],
    proposal_dir: Path,
    truth_dir: Path,
    folds: tuple[Any, ...],
    *,
    run_dir: Path,
    directory_name: str,
    model_name: str,
) -> dict[int, Any]:
    models: dict[int, Any] = {}
    missing: list[Any] = []
    for fold in folds:
        root = run_dir / directory_name / f"fold-{fold.index}"
        model_path = root / model_name
        summary_path = root / "fit-summary.json"
        try:
            models[fold.index] = validator(model_path, summary_path, fold)
            print(f"Reusing {model_name} for fold {fold.index}", flush=True)
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


def _materialize_gap_candidate(
    name: str,
    max_gap_frames: int,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
    complete = _complete_candidate(
        root,
        summary_name="calibration-candidate-summary.json",
        candidate_name=name,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete candidate {name}", flush=True)
        return complete

    source = run_dir / f"{metric._METRIC_GRAPH_NAME}_rts" / "predictions"
    if not source.is_dir():
        raise FileNotFoundError(f"RTS source for gap completion is missing: {source}")
    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    metric.crossfit.evidence._run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.trajectory_gap_completion",
            source,
            "--output-dir",
            output_dir,
            "--output-json",
            root / "gap-completion-summary.json",
            "--max-gap-frames",
            str(max_gap_frames),
            "--max-normalized-speed",
            "5.0",
            "--max-log-size-change",
            "1.0",
            "--min-endpoint-confidence",
            "0.003",
            "--confidence-decay",
            "0.85",
            "--raw-endpoints",
        ],
        log_path=root / "console.txt",
    )
    digest, total_bytes, count = metric.crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    metric.crossfit.evidence._write_json(
        root / "calibration-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-gap-candidate-v1",
            "candidate": name,
            "source_candidate": f"{metric._METRIC_GRAPH_NAME}_rts",
            "max_gap_frames": max_gap_frames,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
        },
    )
    return output_dir


def _materialize_policy(
    source_predictions: Path,
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
    materialize_reference: Callable[..., Path],
    calibration_variants: tuple[tuple[str, tuple[str, ...]], ...],
) -> Path:
    root = run_dir / _POLICY_NAME
    complete = _complete_candidate(
        root,
        summary_name="calibration-candidate-summary.json",
        candidate_name=_POLICY_NAME,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete candidate {_POLICY_NAME}", flush=True)
        return complete

    manifest = metric.crossfit.evidence._load_json(
        run_dir / "proposal-source-manifest.json"
    )
    raw_value = manifest.get("prediction_dir")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("proposal-source manifest does not contain a raw prediction directory")
    raw_predictions = Path(raw_value).expanduser().resolve()
    if not raw_predictions.is_dir():
        raise NotADirectoryError(raw_predictions)

    truth_dir = metric.crossfit._dataset_path_from_inputs(sys.argv[1:], "truth_dir")
    seed_dir = metric.crossfit._dataset_path_from_inputs(
        sys.argv[1:],
        "first_frame_label_dir",
    )
    sequences = metric.crossfit._sequence_manifest(seed_dir, expected_sequences)
    definitions = metric.crossfit._build_fold_definitions(sequences)
    folds = tuple(fold.heldout_sequences for fold in definitions)

    reference_candidates: dict[str, Path] = {"none": raw_predictions}
    target_candidates: dict[str, Path] = {"none": source_predictions}
    for candidate_name, arguments in calibration_variants:
        key = _variant_key(candidate_name)
        reference_name = f"{_REFERENCE_PREFIX}-{key}"
        reference_candidates[key] = materialize_reference(
            raw_predictions,
            reference_name,
            arguments,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        target_path = run_dir / candidate_name / "predictions"
        if not target_path.is_dir():
            raise FileNotFoundError(
                f"scenario policy target candidate is missing: {target_path}"
            )
        target_candidates[key] = target_path

    output_dir = root / "predictions"
    summary = assemble_scenario_policy_predictions(
        reference_candidates,
        target_candidates,
        truth_dir,
        folds,
        output_dir,
    )
    write_summary(summary, root / "scenario-policy-summary.json")
    digest, total_bytes, count = metric.crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"scenario policy covers {count} sequences, expected {expected_sequences}"
        )
    metric.crossfit.evidence._write_json(
        root / "calibration-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-scenario-policy-candidate-v1",
            "candidate": _POLICY_NAME,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "selection_reference": "truth-free raw-control calibration variants",
            "selection_truth_usage": "complementary sequences only",
            "target_predictions": "out-of-fold metric-edge graph variants",
        },
    )
    return output_dir


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


def _clear_terminal_state(run_dir: Path) -> None:
    for path in (
        run_dir / "evidence-summary.json",
        run_dir / "evidence-failure.json",
        run_dir / "tournament" / "tournament_summary.json",
    ):
        path.unlink(missing_ok=True)


def _write_plan(run_dir: Path, requested_run_dir: Path, scope: str) -> None:
    metric.crossfit.evidence._write_json(
        run_dir / "metric-aware-plan.json",
        {
            "schema": _RESUME_SCHEMA,
            "git_sha": os.environ.get("GITHUB_SHA"),
            "legacy_scope": scope,
            "requested_run_dir": str(requested_run_dir),
            "persistent_run_dir": str(run_dir),
            "resume_enabled": run_dir != requested_run_dir,
            "candidate_strategy": (
                "metric-aware ladder only; legacy ladder is evaluated separately"
                if scope == "none"
                else scope
            ),
        },
    )


def main() -> int:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return metric.main()

    scope = _legacy_scope()
    requested_run_dir = _argument_path(sys.argv[1:], "--run-dir")
    persistent_run_dir = _persistent_run_dir(requested_run_dir, scope)
    persistent_run_dir.mkdir(parents=True, exist_ok=True)
    requested_run_dir.mkdir(parents=True, exist_ok=True)
    original_argv = list(sys.argv)
    sys.argv = [
        sys.argv[0],
        *_replace_argument(sys.argv[1:], "--run-dir", persistent_run_dir),
    ]
    _clear_terminal_state(persistent_run_dir)
    _write_plan(persistent_run_dir, requested_run_dir, scope)

    original_variants = metric._CALIBRATION_VARIANTS
    original_extra_names = metric._EXTRA_NAMES
    original_materialize = metric._materialize_calibration_variant
    original_metric_fit = metric._fit_metric_fold_models
    original_metric_candidate = metric._materialize_metric_candidate
    original_legacy_fit = metric.crossfit._fit_fold_models
    original_legacy_candidate = metric.crossfit._materialize_out_of_fold_candidate
    original_score = metric.crossfit.evidence._score
    original_static = metric.crossfit.STATIC_CANDIDATES
    original_oof = metric.crossfit.OUT_OF_FOLD_CANDIDATES
    original_oof_runner = metric.crossfit._run_out_of_fold_candidates

    if scope == "none":
        metric.crossfit.STATIC_CANDIDATES = ()
        metric.crossfit.OUT_OF_FOLD_CANDIDATES = ()

        def no_legacy_candidates(*args: Any, **kwargs: Any) -> dict[str, Path]:
            del args, kwargs
            return {}

        metric.crossfit._run_out_of_fold_candidates = no_legacy_candidates
    elif scope == "best":
        metric.crossfit.STATIC_CANDIDATES = ()
        metric.crossfit.OUT_OF_FOLD_CANDIDATES = tuple(
            candidate
            for candidate in original_oof
            if candidate[0] == "graph_edge_swarm_beam_oof"
        )
        if len(metric.crossfit.OUT_OF_FOLD_CANDIDATES) != 1:
            raise ValueError("legacy best-candidate definition is missing or ambiguous")

    def fit_metric_models(*args: Any, **kwargs: Any) -> dict[int, Any]:
        return _fit_models_resumable(
            original_metric_fit,
            metric._validate_metric_model_provenance,
            *args,
            **kwargs,
            directory_name="cross-fitted-metric-edge-models",
            model_name="metric-edge-model.json",
        )

    def fit_legacy_models(*args: Any, **kwargs: Any) -> dict[int, Any]:
        return _fit_models_resumable(
            original_legacy_fit,
            metric.crossfit._validate_model_provenance,
            *args,
            **kwargs,
            directory_name="cross-fitted-edge-models",
            model_name="edge-model.json",
        )

    def materialize_metric_candidate(*args: Any, **kwargs: Any) -> Path:
        expected = int(kwargs["expected_sequences"])
        run_dir = Path(kwargs["run_dir"])
        complete = _complete_candidate(
            run_dir / metric._METRIC_GRAPH_NAME,
            summary_name="cross-fitted-candidate-summary.json",
            candidate_name=metric._METRIC_GRAPH_NAME,
            expected_sequences=expected,
        )
        if complete is not None:
            print(
                f"Reusing complete candidate {metric._METRIC_GRAPH_NAME}",
                flush=True,
            )
            return complete
        return original_metric_candidate(*args, **kwargs)

    def materialize_legacy_candidate(*args: Any, **kwargs: Any) -> Path:
        name = str(args[0])
        expected = int(kwargs["expected_sequences"])
        run_dir = Path(kwargs["run_dir"])
        complete = _complete_candidate(
            run_dir / name,
            summary_name="cross-fitted-candidate-summary.json",
            candidate_name=name,
            expected_sequences=expected,
        )
        if complete is not None:
            print(f"Reusing complete candidate {name}", flush=True)
            return complete
        return original_legacy_candidate(*args, **kwargs)

    def score(*args: Any, **kwargs: Any) -> dict[str, Any]:
        name = str(args[0])
        expected = int(kwargs["expected_sequences"])
        run_dir = Path(kwargs["run_dir"])
        metrics_path = run_dir / name / "metrics.json"
        prediction_dir = Path(args[1])
        if metrics_path.is_file() and _scorecard_is_fresh(
            metrics_path,
            prediction_dir,
        ):
            try:
                payload = _load_json(metrics_path)
                if int(payload.get("sequence_count", -1)) == expected:
                    print(f"Reusing scorecard for {name}", flush=True)
                    return payload
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return original_score(*args, **kwargs)

    metric._fit_metric_fold_models = fit_metric_models
    metric._materialize_metric_candidate = materialize_metric_candidate
    metric.crossfit._fit_fold_models = fit_legacy_models
    metric.crossfit._materialize_out_of_fold_candidate = materialize_legacy_candidate
    metric.crossfit.evidence._score = score

    gap_placeholders = tuple((name, ()) for name, _max_gap in _GAP_VARIANTS)
    metric._CALIBRATION_VARIANTS = (
        *original_variants,
        *gap_placeholders,
        (_POLICY_NAME, ()),
    )
    metric._EXTRA_NAMES = (
        *original_extra_names,
        *(name for name, _max_gap in _GAP_VARIANTS),
        _POLICY_NAME,
    )
    gap_lookup = dict(_GAP_VARIANTS)

    def materialize_base_candidate(
        source_predictions: Path,
        name: str,
        arguments: tuple[str, ...],
        resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
        *,
        run_dir: Path,
        expected_sequences: int,
    ) -> Path:
        complete = _complete_candidate(
            run_dir / name,
            summary_name="calibration-candidate-summary.json",
            candidate_name=name,
            expected_sequences=expected_sequences,
        )
        if complete is not None:
            print(f"Reusing complete candidate {name}", flush=True)
            return complete
        output_dir = original_materialize(
            source_predictions,
            name,
            arguments,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        if name.startswith(f"{_REFERENCE_PREFIX}-"):
            summary_path = run_dir / name / "calibration-candidate-summary.json"
            summary = _load_json(summary_path)
            summary["source_candidate"] = "raw"
            summary["selection_role"] = "scenario-policy reference"
            metric.crossfit.evidence._write_json(summary_path, summary)
        return output_dir

    def materialize(
        source_predictions: Path,
        name: str,
        arguments: tuple[str, ...],
        resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
        *,
        run_dir: Path,
        expected_sequences: int,
    ) -> Path:
        if name in gap_lookup:
            return _materialize_gap_candidate(
                name,
                gap_lookup[name],
                run_dir=run_dir,
                expected_sequences=expected_sequences,
            )
        if name == _POLICY_NAME:
            return _materialize_policy(
                source_predictions,
                resolution_groups,
                run_dir=run_dir,
                expected_sequences=expected_sequences,
                materialize_reference=materialize_base_candidate,
                calibration_variants=original_variants,
            )
        return materialize_base_candidate(
            source_predictions,
            name,
            arguments,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )

    metric._materialize_calibration_variant = materialize
    try:
        return metric.main()
    finally:
        metric._CALIBRATION_VARIANTS = original_variants
        metric._EXTRA_NAMES = original_extra_names
        metric._materialize_calibration_variant = original_materialize
        metric._fit_metric_fold_models = original_metric_fit
        metric._materialize_metric_candidate = original_metric_candidate
        metric.crossfit._fit_fold_models = original_legacy_fit
        metric.crossfit._materialize_out_of_fold_candidate = original_legacy_candidate
        metric.crossfit.evidence._score = original_score
        metric.crossfit.STATIC_CANDIDATES = original_static
        metric.crossfit.OUT_OF_FOLD_CANDIDATES = original_oof
        metric.crossfit._run_out_of_fold_candidates = original_oof_runner
        sys.argv = original_argv
        _sync_compact_evidence(persistent_run_dir, requested_run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
