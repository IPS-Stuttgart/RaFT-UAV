# Reliability-prioritized MMUAD anchor coverage

The MMUAD candidate-assignment experiments show that useful physical hypotheses
can remain deeper in the candidate pool even after posterior-mass group
selection. Multi-anchor coverage rescue protects those modes, but a finite
rescue budget can still be spent on whichever anchor is processed first.

This experiment combines reliability-weighted anchor-cost quantile selection
with a global framewise rescue ranking. Every positive-weight anchor proposes
its nearest supported physical group. Proposals for the same group are merged,
then the bounded rescue budget is allocated by one of three inference-safe
priorities:

- `distance`: nearest supported group first;
- `reliability`: largest summed anchor reliability first;
- `reliability-distance`: largest
  `sum(weight / (1 + distance / distance_scale))` first.

The final grouped learned-sigma / Huber mixture-MAP objective is unchanged.
Ground truth is not used for candidate selection or rescue. Anchor reliability
weights and all rescue hyperparameters must be selected on training folds and
then frozen before validation or hidden-test inference.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_weighted_anchor_coverage.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv robust=/path/to/robust_estimates.csv \
  --anchor-csv score_top1=/path/to/score_top1_estimates.csv \
  --anchor-csv frame_median=/path/to/frame_median_estimates.csv \
  --anchor-reliability robust=8 \
  --anchor-reliability score_top1=2 \
  --anchor-reliability frame_median=1 \
  --anchor-cost-quantile 0.5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --anchor-coverage-max-distance-m 25 \
  --anchor-coverage-max-extra-groups-per-frame 2 \
  --anchor-coverage-priority-mode reliability-distance \
  --anchor-coverage-distance-scale-m 10 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_weighted_anchor_priority_coverage
```

## Suggested train-CV ablation

Freeze the candidate pool, learned sigma, Huber mixture objective, physical
hypothesis grouping, anchor set, and final initialization. Select only:

```text
anchor_cost_quantile:                    0.25, 0.5, 0.75
priority_mode:                           distance, reliability,
                                         reliability-distance
max_anchor_distance_m:                  5, 10, 20, 30
max_extra_groups_per_frame:             0, 1, 2, 3
max_siblings_per_rescued_group:         1, 2
coverage_distance_scale_m:              2, 5, 10, 20
```

Compare pose MSE, P95, physical-group top-K oracle recall, rescued-group rate,
covered anchor reliability, blocked reliability, candidate count, and runtime.
The main diagnostic question is whether the finite rescue budget protects modes
supported by the most reliable anchors rather than modes encountered first.
