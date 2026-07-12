# Reliability-weighted multi-anchor MMUAD selection

The multi-anchor posterior-mass selector protects candidate modes that agree
with any inference-time trajectory anchor. That improves physical-hypothesis
recall, but it gives a weak anchor the same influence as a strong one.
Reliability-weighted aggregation adds a train-selectable non-negative weight per
anchor without changing the final grouped mixture-MAP objective.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_weighted_multi_anchor_mass_topk.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv robust=/path/to/robust_estimates.csv \
  --anchor-csv score_top1=/path/to/score_top1_estimates.csv \
  --anchor-csv frame_median=/path/to/frame_median_estimates.csv \
  --anchor-reliability robust=4 \
  --anchor-reliability score_top1=2 \
  --anchor-reliability frame_median=1 \
  --aggregation softmin \
  --softmin-temperature 0.5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_weighted_multi_anchor
```

Unspecified anchors use `--default-anchor-reliability`, which defaults to
`1.0`. Weights must be finite and non-negative, and at least one anchor must
have positive weight.

## Aggregation semantics

- `mean`: weighted arithmetic mean of available anchor costs.
- `softmin`: reliability-weighted log-sum-exp soft minimum.
- `minimum`: minimum over positive-weight anchors; weight magnitude is ignored,
  while a zero weight excludes an anchor.

Weights are renormalized over the anchors that match each candidate timestamp.
The output records matched anchor weight and the effective number of anchors so
coverage differences are visible.

## Train-CV experiment

Freeze the candidate reservoir, learned sigma, physical grouping, robust Huber
mixture objective, and final initialization. Select anchor weights on train
only, for example:

```text
anchor sets:
  robust + score_top1
  robust + score_top1 + frame_median
  robust + branch starts

relative weights:
  1:1
  2:1
  4:2:1
  8:2:1

aggregation:
  mean
  softmin

softmin temperature:
  0.1, 0.25, 0.5, 1.0
```

Compare pose MSE, P95, physical-group top-K oracle recall, selected group
budget, best-anchor counts, matched weight, and effective anchor count. Ground
truth may be used to select weights in train CV, but is never read by the
inference-time selector.
