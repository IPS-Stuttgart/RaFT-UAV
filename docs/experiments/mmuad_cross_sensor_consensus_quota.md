# MMUAD cross-sensor consensus quota

The branch-preserving reservoir protects candidate-generation provenance, but
its source and branch quotas remain unary-score driven. A low-scoring candidate
can therefore disappear even when another independent sensor supports nearly
the same 3D position.

`candidate_consensus_quota.py` computes the existing truth-free branch-consensus
features on the complete candidate pool and protects a small number of
cross-source-supported candidates before the final frame cap.

The quota:

- requires configurable cross-source neighbour and unique-source counts;
- limits the nearest supporting-sensor distance;
- excludes raw/calibrated siblings from supporting one another by default;
- limits how many coordinate representations of one physical origin can consume
  the quota; and
- guarantees selected consensus rows survive the final candidate cap.

Truth is never used for selection. Optional truth input writes oracle-recall
artifacts only.

## Example

```bash
python scripts/mmuad_candidate_consensus_quota.py \
  --candidates-csv outputs/mmuad_candidates/full_branch_pool.csv \
  --output-reservoir-csv outputs/mmuad_consensus_quota/reservoir.csv \
  --summary-json outputs/mmuad_consensus_quota/summary.json \
  --score-column candidate_risk_adjusted_score \
  --fallback-score-column ranker_score \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --consensus-top-n 2 \
  --min-neighbor-count 1 \
  --min-unique-source-count 1 \
  --max-nearest-distance-m 5 \
  --max-per-origin 1
```

## Train-CV ablation

Select all controls on training folds, freeze them, then evaluate public
validation once.

```text
consensus_top_n:             0, 1, 2, 3, 5
max_nearest_distance_m:      1, 2, 5, 10
min_unique_source_count:     1, 2
max_per_origin:              1, 2
max_per_source:              0, 1, 2
consensus time window:       0.02, 0.05, 0.10 s
reservoir frame cap:         20, 40, 60
```

Compare:

1. full/top-3/top-5/top-10/top-20 oracle recall;
2. physical-origin-group oracle recall;
3. frozen learned-sigma Huber mixture-MAP pose MSE and P95; and
4. the number of quota rows that displace ordinary branch/source candidates.

The experiment is successful only when train-CV selects a nonzero quota and the
frozen validation run improves pose or materially improves candidate recall
without increasing catastrophic-tail error.
