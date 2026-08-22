#!/usr/bin/env python3
"""Run leakage-safe out-of-fold learned-edge evidence for Multi-UAV LTS."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_multi_uav_lts_improved_evidence as improved
import run_multi_uav_lts_public_evidence as evidence
from raft_uav.multi_uav_lts.fixed_population_cv import build_stratified_folds

FOLD_COUNT = 5
FOLD_SEED = 0
_EDGE_MODEL_TOKEN = "{EDGE_MODEL_JSON}"

_DELAYED_COMMON_MOTION = (
    "--enable-delayed-path-cover",
    "--delayed-max-gap",
    "0",
    "--delayed-lookahead-frames",
    "2",
    "--delayed-successors-per-frame",
    "3",
    "--delayed-continuation-weight",
    "0.75",
    "--enable-common-motion",
    "--common-motion-min-pairs",
    "4",
    "--common-motion-max-normalized-step",
    "8.0",
    "--common-motion-max-normalized-residual",
    "1.5",
)
_SIMILARITY_MOTION = (
    "--common-motion-model",
    "similarity",
    "--similarity-min-pairs",
    "4",
    "--similarity-max-scale-change",
    "0.12",
    "--similarity-max-rotation-deg",
    "10.0",
    "--similarity-max-normalized-residual",
    "1.0",
    "--similarity-min-normalized-spread",
    "2.0",
    "--similarity-min-residual-improvement",
    "0.05",
)
_SWARM_RELATIVE = (
    "--swarm-relative-weight",
    "0.75",
    "--swarm-relative-clip",
    "4.0",
    "--swarm-neighbors",
    "4",
    "--swarm-radius-scale",
    "12.0",
)
_EDGE_MODEL = (
    "--edge-model-json",
    _EDGE_MODEL_TOKEN,
    "--edge-model-weight",
    "1.0",
    "--edge-model-clip",
    "4.0",
)
_AMBIGUITY_BEAM = (
    "--enable-ambiguity-beam",
    "--ambiguity-beam-width",
    "8",
    "--ambiguity-beam-max-component-nodes",
    "16",
    "--ambiguity-beam-margin",
    "1.0",
    "--ambiguity-acceleration-weight",
    "1.0",
    "--ambiguity-acceleration-clip",
    "4.0",
)

STATIC_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("graph_delayed_translation", _DELAYED_COMMON_MOTION),
    (
        "graph_delayed_similarity",
        (*_DELAYED_COMMON_MOTION, *_SIMILARITY_MOTION),
    ),
    (
        "graph_delayed_swarm",
        (*_DELAYED_COMMON_MOTION, *_SWARM_RELATIVE),
    ),
)

OUT_OF_FOLD_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "graph_edge_oof",
        (*_DELAYED_COMMON_MOTION, *_EDGE_MODEL),
    ),
    (
        "graph_edge_swarm_oof",
        (*_DELAYED_COMMON_MOTION, *_EDGE_MODEL, *_SWARM_RELATIVE),
    ),
    (
        "graph_edge_swarm_beam_oof",
        (
            *_DELAYED_COMMON_MOTION,
            *_EDGE_MODEL,
            *_SWARM_RELATIVE,
            *_AMBIGUITY_BEAM,
        ),
    ),
)


@dataclass(frozen=True)
class FoldDefinition:
    """One held-out fold and the complementary edge-model training panel."""

    index: int
    training_sequences: tuple[str, ...]
    heldout_sequences: tuple[str, ...]


@dataclass(frozen=True)
class FoldModel:
    """Validated edge model fitted without its held-out sequences."""

    fold_index: int
    model_path: Path
    summary_path: Path
    model_sha256: str
    training_sequences: tuple[str, ...]
    heldout_sequences: tuple[str, ...]


def _dataset_path_from_inputs(arguments: Sequence[str], field: str) -> Path:
    payload_path = improved._inputs_json_path(arguments)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    try:
        raw_path = payload["dataset"][field]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{payload_path}: missing dataset.{field}") from exc
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{payload_path}: invalid dataset.{field}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"dataset.{field} is not a directory: {path}")
    return path


def _sequence_manifest(seed_dir: Path, expected_sequences: int) -> tuple[str, ...]:
    sequences = tuple(sorted(path.stem for path in seed_dir.glob("*.txt")))
    if len(sequences) != expected_sequences:
        raise ValueError(
            f"seed manifest covers {len(sequences)} sequences, "
            f"expected {expected_sequences}"
        )
    if len(set(sequences)) != len(sequences):
        raise ValueError("seed manifest contains duplicate sequence names")
    return sequences


def _build_fold_definitions(
    sequences: tuple[str, ...],
    *,
    fold_count: int = FOLD_COUNT,
    seed: int = FOLD_SEED,
) -> tuple[FoldDefinition, ...]:
    folds = build_stratified_folds(sequences, fold_count=fold_count, seed=seed)
    sequence_set = set(sequences)
    seen: set[str] = set()
    definitions: list[FoldDefinition] = []
    for index, heldout in enumerate(folds):
        heldout_set = set(heldout)
        if not heldout or len(heldout_set) != len(heldout):
            raise ValueError(f"fold {index} is empty or contains duplicates")
        overlap = seen & heldout_set
        if overlap:
            raise ValueError(
                f"sequences occur in multiple held-out folds: {sorted(overlap)}"
            )
        training = tuple(sequence for sequence in sequences if sequence not in heldout_set)
        if not training or set(training) & heldout_set:
            raise ValueError(f"fold {index} has an invalid training complement")
        definitions.append(
            FoldDefinition(
                index=index,
                training_sequences=training,
                heldout_sequences=tuple(heldout),
            )
        )
        seen.update(heldout_set)
    if seen != sequence_set:
        raise ValueError(
            "held-out fold union differs from the sequence manifest: "
            f"missing={sorted(sequence_set - seen)}, extra={sorted(seen - sequence_set)}"
        )
    return tuple(definitions)


def _selected_resolution_groups(
    groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    sequences: tuple[str, ...],
) -> tuple[tuple[tuple[int, int], tuple[str, ...]], ...]:
    selected = set(sequences)
    result: list[tuple[tuple[int, int], tuple[str, ...]]] = []
    covered: set[str] = set()
    for dimensions, group_sequences in groups:
        subset = tuple(sequence for sequence in group_sequences if sequence in selected)
        if not subset:
            continue
        overlap = covered & set(subset)
        if overlap:
            raise ValueError(f"resolution inventory duplicates sequences: {sorted(overlap)}")
        result.append((dimensions, subset))
        covered.update(subset)
    if covered != selected:
        raise ValueError(
            "resolution inventory does not cover the requested fold: "
            f"missing={sorted(selected - covered)}, extra={sorted(covered - selected)}"
        )
    return tuple(result)


def _resolve_edge_model_arguments(
    arguments: tuple[str, ...],
    model_path: Path,
) -> tuple[str, ...]:
    if arguments.count(_EDGE_MODEL_TOKEN) != 1:
        raise ValueError("candidate arguments must contain one edge-model placeholder")
    return tuple(
        str(model_path) if argument == _EDGE_MODEL_TOKEN else argument
        for argument in arguments
    )


def _validate_model_provenance(
    model_path: Path,
    summary_path: Path,
    fold: FoldDefinition,
) -> FoldModel:
    model = evidence._load_json(model_path)
    summary = evidence._load_json(summary_path)
    expected_training = list(fold.training_sequences)
    selected_summary = summary.get("selected_sequences")
    metadata = model.get("metadata")
    selected_model = metadata.get("selected_sequences") if isinstance(metadata, Mapping) else None
    if selected_summary != expected_training or selected_model != expected_training:
        raise ValueError(
            f"fold {fold.index} model sequence provenance does not match its training panel"
        )
    if set(fold.heldout_sequences) & set(expected_training):
        raise ValueError(f"fold {fold.index} training panel leaks held-out sequences")
    if model.get("sequence_count") != len(expected_training):
        raise ValueError(f"fold {fold.index} model sequence_count is inconsistent")
    if summary.get("selected_sequence_count") != len(expected_training):
        raise ValueError(f"fold {fold.index} fit summary sequence count is inconsistent")
    if int(summary.get("positive_candidate_edges", 0)) <= 0:
        raise ValueError(f"fold {fold.index} model has no positive training edges")
    if int(summary.get("negative_candidate_edges", 0)) <= 0:
        raise ValueError(f"fold {fold.index} model has no negative training edges")
    return FoldModel(
        fold_index=fold.index,
        model_path=model_path,
        summary_path=summary_path,
        model_sha256=evidence._sha256(model_path),
        training_sequences=fold.training_sequences,
        heldout_sequences=fold.heldout_sequences,
    )


def _fit_fold_models(
    proposal_dir: Path,
    truth_dir: Path,
    folds: tuple[FoldDefinition, ...],
    *,
    run_dir: Path,
) -> dict[int, FoldModel]:
    models: dict[int, FoldModel] = {}
    for fold in folds:
        root = run_dir / "cross-fitted-edge-models" / f"fold-{fold.index}"
        model_path = root / "edge-model.json"
        summary_path = root / "fit-summary.json"
        evidence._run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.proposal_edge_model",
                proposal_dir,
                "--truth-dir",
                truth_dir,
                "--output-json",
                model_path,
                "--summary-json",
                summary_path,
                "--max-gap",
                "0",
                "--max-link-cost",
                "2.25",
                "--negative-candidates-per-left",
                "5",
                "--enable-common-motion",
                "--common-motion-min-pairs",
                "4",
                "--common-motion-max-normalized-step",
                "8.0",
                "--common-motion-max-normalized-residual",
                "1.5",
                "--swarm-neighbors",
                "4",
                "--swarm-radius-scale",
                "12.0",
                "--swarm-unmatched-penalty",
                "2.0",
                "--l2-penalty",
                "1.0",
                "--sequences",
                *fold.training_sequences,
            ],
            log_path=root / "fit-console.txt",
        )
        models[fold.index] = _validate_model_provenance(
            model_path,
            summary_path,
            fold,
        )
    return models


def _run_tracker_group(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    output_dir: Path,
    summary_path: Path,
    log_path: Path,
    dimensions: tuple[int, int],
    arguments: tuple[str, ...],
    sequences: tuple[str, ...],
) -> None:
    width, height = dimensions
    evidence._run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.experimental_proposal_graph_tracker",
            proposal_dir,
            "--first-frame-label-dir",
            seed_dir,
            "--output-dir",
            output_dir,
            "--output-json",
            summary_path,
            "--image-width",
            str(width),
            "--image-height",
            str(height),
            *arguments,
            "--sequences",
            *sequences,
        ],
        log_path=log_path,
    )


def _materialize_out_of_fold_candidate(
    name: str,
    arguments: tuple[str, ...],
    proposal_dir: Path,
    seed_dir: Path,
    folds: tuple[FoldDefinition, ...],
    models: Mapping[int, FoldModel],
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    copied_names: set[str] = set()
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        model = models.get(fold.index)
        if model is None:
            raise ValueError(f"missing fitted model for fold {fold.index}")
        if model.heldout_sequences != fold.heldout_sequences:
            raise ValueError(f"fold {fold.index} model was assigned to the wrong fold")
        fold_arguments = _resolve_edge_model_arguments(arguments, model.model_path)
        group_records: list[dict[str, Any]] = []
        for dimensions, sequences in _selected_resolution_groups(
            resolution_groups,
            fold.heldout_sequences,
        ):
            width, height = dimensions
            group_name = f"fold-{fold.index}-{width}x{height}"
            group_root = root / "fold-groups" / group_name
            group_output = group_root / "predictions"
            shutil.rmtree(group_output, ignore_errors=True)
            _run_tracker_group(
                proposal_dir,
                seed_dir,
                output_dir=group_output,
                summary_path=group_root / "summary.json",
                log_path=group_root / "console.txt",
                dimensions=dimensions,
                arguments=fold_arguments,
                sequences=sequences,
            )
            _digest, total_bytes, count = evidence._directory_digest(group_output)
            if count != len(sequences) or total_bytes <= 0:
                raise ValueError(
                    f"{name}/{group_name} produced {count} nonempty files for "
                    f"{len(sequences)} requested sequences"
                )
            for source in sorted(group_output.glob("*.txt")):
                if source.name in copied_names:
                    raise ValueError(f"{name}: duplicate held-out prediction {source.name}")
                shutil.copy2(source, output_dir / source.name)
                copied_names.add(source.name)
            group_records.append(
                {
                    "width": width,
                    "height": height,
                    "sequences": list(sequences),
                    "prediction_content_bytes": total_bytes,
                }
            )
        fold_records.append(
            {
                "fold": fold.index,
                "training_sequences": list(fold.training_sequences),
                "heldout_sequences": list(fold.heldout_sequences),
                "edge_model_path": str(model.model_path),
                "edge_model_sha256": model.model_sha256,
                "resolution_groups": group_records,
            }
        )

    expected_names = {f"{sequence}.txt" for fold in folds for sequence in fold.heldout_sequences}
    if copied_names != expected_names:
        raise ValueError(
            f"{name} held-out assembly is incomplete: "
            f"missing={sorted(expected_names - copied_names)}, "
            f"extra={sorted(copied_names - expected_names)}"
        )
    digest, total_bytes, count = evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    evidence._write_json(
        root / "cross-fitted-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-cross-fitted-edge-candidate-v1",
            "candidate": name,
            "candidate_arguments_template": list(arguments),
            "fold_count": len(folds),
            "fold_seed": FOLD_SEED,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "truth_usage": "edge models only; every prediction uses a complementary fold",
            "folds": fold_records,
        },
    )
    return output_dir


def _run_out_of_fold_candidates(
    proposal_dir: Path,
    seed_dir: Path,
    truth_dir: Path,
    *,
    image_root: Path,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    sequences = _sequence_manifest(seed_dir, expected_sequences)
    folds = _build_fold_definitions(sequences)
    resolution_groups = improved._sequence_resolution_groups(image_root, seed_dir)
    models = _fit_fold_models(proposal_dir, truth_dir, folds, run_dir=run_dir)
    evidence._write_json(
        run_dir / "cross-fitted-edge-folds.json",
        {
            "schema": "raft-uav-multi-uav-lts-cross-fitted-edge-folds-v1",
            "fold_count": FOLD_COUNT,
            "fold_seed": FOLD_SEED,
            "sequence_count": len(sequences),
            "sequences": list(sequences),
            "folds": [
                {
                    "fold": fold.index,
                    "training_sequences": list(fold.training_sequences),
                    "heldout_sequences": list(fold.heldout_sequences),
                    "edge_model_sha256": models[fold.index].model_sha256,
                }
                for fold in folds
            ],
        },
    )
    return {
        name: _materialize_out_of_fold_candidate(
            name,
            arguments,
            proposal_dir,
            seed_dir,
            folds,
            models,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        for name, arguments in OUT_OF_FOLD_CANDIDATES
    }


def _available_results_with_out_of_fold(
    original,
    run_dir: Path,
) -> dict[str, Any]:
    payload = original(run_dir)
    metrics = payload.setdefault("metrics", {})
    for name, _arguments in OUT_OF_FOLD_CANDIDATES:
        path = run_dir / name / "metrics.json"
        if not path.is_file():
            continue
        try:
            metrics[name] = evidence._load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            metrics[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    return payload


def main() -> int:
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return improved.main()
    truth_dir = _dataset_path_from_inputs(sys.argv[1:], "truth_dir")
    original_candidates = evidence.CANDIDATES
    original_improved_candidates = improved.IMPROVED_CANDIDATES
    original_runner = improved._run_candidates_with_native_dimensions
    original_available_results = evidence._available_results
    evidence.CANDIDATES = ()
    improved.IMPROVED_CANDIDATES = STATIC_CANDIDATES

    def run_candidates(
        proposal_dir: Path,
        seed_dir: Path,
        *,
        image_root: Path,
        run_dir: Path,
        expected_sequences: int,
    ) -> dict[str, Path]:
        static_outputs = original_runner(
            proposal_dir,
            seed_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        out_of_fold_outputs = _run_out_of_fold_candidates(
            proposal_dir,
            seed_dir,
            truth_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        overlap = set(static_outputs) & set(out_of_fold_outputs)
        if overlap:
            raise ValueError(f"duplicate candidate names: {sorted(overlap)}")
        return {**static_outputs, **out_of_fold_outputs}

    def available_results(run_dir: Path) -> dict[str, Any]:
        return _available_results_with_out_of_fold(original_available_results, run_dir)

    improved._run_candidates_with_native_dimensions = run_candidates
    evidence._available_results = available_results
    try:
        return improved.main()
    finally:
        evidence.CANDIDATES = original_candidates
        improved.IMPROVED_CANDIDATES = original_improved_candidates
        improved._run_candidates_with_native_dimensions = original_runner
        evidence._available_results = original_available_results


if __name__ == "__main__":
    raise SystemExit(main())
