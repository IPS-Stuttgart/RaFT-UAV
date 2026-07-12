# MMUAD multi-anchor physical-group coverage

The multi-anchor posterior-mass selector protects candidates that agree with any
inference-time trajectory anchor. A fixed finite group budget can nevertheless
retain only one anchor-supported mode when several plausible trajectories are
far apart.

`candidate_mixture_group_multi_anchor_coverage.py` adds a bounded rescue after
posterior-mass group selection:

1. run the maintained multi-anchor score/uncertainty selector;
2. identify the nearest physical hypothesis group to every supported anchor;
3. retain a missing group when it is within a configured distance;
4. cap the number of rescued groups and siblings per frame;
5. run the unchanged grouped learned-sigma / Huber mixture-MAP objective.

Ground truth is never used for selection. It remains optional for downstream
metrics only.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_multi_anchor_coverage.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv robust=/path/to/robust_estimates.csv \
  --anchor-csv score_top1=/path/to/score_top1_estimates.csv \
  --anchor-csv translated=/path/to/translated_estimates.csv \
  --aggregation minimum \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --min-group-top-k 3 \
  --max-group-top-k 10 \
  --target-posterior-mass 0.95 \
  --anchor-coverage-max-distance-m 25 \
  --anchor-coverage-max-extra-groups-per-frame 2 \
  --anchor-coverage-max-siblings-per-rescued-group 1 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_multi_anchor_group_coverage
```

## Train-CV ablation

Freeze the candidate pool, score model, learned sigma, Huber objective, and
anchor trajectories. Select only:

```text
max_anchor_distance_m:            5, 10, 20, 30, 50
max_extra_groups_per_frame:       0, 1, 2, 3
max_siblings_per_rescued_group:   1, 2
anchor aggregation:               minimum, softmin
```

Compare pose MSE, P95, physical-group top-K oracle recall, rescued-group rate,
frames hitting the rescue budget, and runtime. The `max_extra_groups_per_frame=0`
row is the exact no-rescue control.
