"""Global seeded-label multi-scan association by dual-decomposed path optimization.

Each supplied frame-one identity receives a sequence-level path through the
proposal DAG. Shared proposal conflicts are resolved by Lagrange prices. Late
births are recovered only from persistent unclaimed paths.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
import numpy as np
from ._full_stack_io import prediction_files, read_rows, write_json, write_rows
from ._records import Detection

@dataclass(frozen=True)
class MultiScanConfig:
    max_gap: int = 12
    motion_scale: float = 5.0
    size_scale: float = 0.7
    miss_cost: float = 0.22
    confidence_reward: float = 0.8
    start_gate: float = 8.0
    dual_iterations: int = 16
    dual_step: float = 0.8
    min_birth_hits: int = 5
    min_birth_span: int = 4
    min_birth_confidence: float = 0.01
    affinity_weight: float = 0.0

def edge_cost(a: Detection, b: Detection, config: MultiScanConfig, affinity: Callable[[Detection, Detection], float] | None=None) -> float:
    gap = b.frame_id - a.frame_id
    if gap <= 0 or gap > config.max_gap:
        return math.inf
    scale = max(math.sqrt(a.width * a.height), math.sqrt(b.width * b.height), 2.0)
    distance = math.hypot(b.center_x - a.center_x, b.center_y - a.center_y) / (scale * math.sqrt(gap))
    size = abs(math.log(max(b.width, 1e-06) / max(a.width, 1e-06))) + abs(math.log(max(b.height, 1e-06) / max(a.height, 1e-06)))
    cost = (distance / config.motion_scale) ** 2 + (size / config.size_scale) ** 2 + config.miss_cost * (gap - 1) - config.confidence_reward * math.log(max(b.confidence, 1e-05) / 1e-05) / math.log(100000.0)
    if affinity is not None:
        cost -= config.affinity_weight * float(affinity(a, b))
    return cost

def _best_path(anchor: Detection, nodes: list[Detection], prices: dict[tuple[int, int], float], config: MultiScanConfig, affinity=None, forbid_start: bool=False) -> list[Detection]:
    ordered = sorted(nodes, key=lambda r: (r.frame_id, r.object_id))
    costs = np.full(len(ordered), math.inf)
    parent = np.full(len(ordered), -1, dtype=int)
    for i, node in enumerate(ordered):
        start = edge_cost(anchor, node, config, affinity)
        if not forbid_start and math.isfinite(start):
            costs[i] = start + prices.get((node.frame_id, node.object_id), 0.0)
        for j in range(i):
            if not math.isfinite(costs[j]):
                continue
            c = edge_cost(ordered[j], node, config, affinity)
            if math.isfinite(c) and costs[j] + c + prices.get((node.frame_id, node.object_id), 0.0) < costs[i]:
                costs[i] = costs[j] + c + prices.get((node.frame_id, node.object_id), 0.0)
                parent[i] = j
    if not len(ordered) or not np.isfinite(costs).any():
        return [anchor]
    terminal = costs - np.asarray([config.miss_cost * 0.5 for _ in ordered])
    end = int(np.argmin(terminal))
    path = []
    while end >= 0:
        path.append(ordered[end])
        end = int(parent[end])
    return [anchor, *reversed(path)]

def associate_sequence(proposals: list[Detection], seeds: list[Detection], config: MultiScanConfig=MultiScanConfig(), affinity=None) -> tuple[list[Detection], dict]:
    nodes = [p for p in proposals if p.frame_id > 1]
    prices = {}
    paths = {seed.object_id: [seed] for seed in seeds}
    for iteration in range(config.dual_iterations):
        paths = {seed.object_id: _best_path(seed, nodes, prices, config, affinity) for seed in seeds}
        usage = {}
        for path in paths.values():
            for node in path[1:]:
                usage.setdefault((node.frame_id, node.object_id), []).append(node)
        conflicts = {key: values for key, values in usage.items() if len(values) > 1}
        if not conflicts:
            break
        step = config.dual_step / math.sqrt(iteration + 1)
        for key, values in conflicts.items():
            prices[key] = prices.get(key, 0) + step * (len(values) - 1)
    claims = {}
    for label, path in paths.items():
        seed = next((s for s in seeds if s.object_id == label))
        for node in path[1:]:
            key = (node.frame_id, node.object_id)
            score = edge_cost(seed, node, config, affinity)
            if key not in claims or score < claims[key][0]:
                claims[key] = (score, label, node)
    output = []
    claimed = set()
    for seed in seeds:
        output.append(seed)
        for node in paths.get(seed.object_id, ())[1:]:
            key = (node.frame_id, node.object_id)
            if claims.get(key, (None, None, None))[1] == seed.object_id:
                output.append(replace(node, object_id=seed.object_id))
                claimed.add(key)
    remaining = [n for n in nodes if (n.frame_id, n.object_id) not in claimed]
    next_id = max([s.object_id for s in seeds], default=0) + 1
    births = 0
    while remaining:
        anchor = min(remaining, key=lambda r: (r.frame_id, -r.confidence, r.object_id))
        pseudo = replace(anchor, frame_id=max(1, anchor.frame_id - 1))
        path = _best_path(pseudo, [n for n in remaining if n is not anchor], prices, config, affinity)
        chain = [anchor, *path[1:]]
        mean = float(np.mean([n.confidence for n in chain]))
        span = chain[-1].frame_id - chain[0].frame_id
        if len(chain) >= config.min_birth_hits and span >= config.min_birth_span and (mean >= config.min_birth_confidence):
            output.extend((replace(n, object_id=next_id) for n in chain))
            next_id += 1
            births += 1
            keys = {(n.frame_id, n.object_id) for n in chain}
            remaining = [n for n in remaining if (n.frame_id, n.object_id) not in keys]
        else:
            remaining.remove(anchor)
    diagnostics = {'seed_count': len(seeds), 'proposal_count': len(proposals), 'output_count': len(output), 'late_birth_count': births, 'conflict_price_count': len(prices)}
    return (output, diagnostics)

def associate_directory(proposal_dir: Path, seed_dir: Path, output_dir: Path, sequence_names: list[str] | None=None, config: MultiScanConfig=MultiScanConfig(), affinity=None) -> dict:
    allowed = None if sequence_names is None else set(sequence_names)
    summary = {}
    for proposal_file in prediction_files(proposal_dir):
        name = proposal_file.stem
        if allowed is not None and name not in allowed:
            continue
        seed_file = seed_dir / proposal_file.name
        if not seed_file.is_file():
            raise FileNotFoundError(seed_file)
        rows, diag = associate_sequence(read_rows(proposal_file), read_rows(seed_file), config, affinity)
        write_rows(output_dir / proposal_file.name, rows)
        summary[name] = diag
    return {'sequences': summary, 'config': config.__dict__}

def _names(path):
    return None if path is None else [x.strip() for x in path.read_text().splitlines() if x.strip()]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('proposal_dir', type=Path)
    p.add_argument('--seed-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sequence-list', type=Path)
    p.add_argument('--max-gap', type=int, default=12)
    p.add_argument('--dual-iterations', type=int, default=16)
    p.add_argument('--min-birth-hits', type=int, default=5)
    p.add_argument('--summary-json', type=Path)
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    config = MultiScanConfig(max_gap=a.max_gap, dual_iterations=a.dual_iterations, min_birth_hits=a.min_birth_hits)
    result = associate_directory(a.proposal_dir, a.seed_dir, a.output_dir, _names(a.sequence_list), config)
    if a.summary_json:
        write_json(a.summary_json, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
