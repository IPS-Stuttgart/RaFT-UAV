# MMUAD reliability-quantile anchor aggregation

The multi-anchor physical-group selector can use either a permissive minimum
anchor cost or an average/soft minimum. These choices have complementary
failure modes:

- `minimum` preserves a physical mode supported by any anchor, including a weak
  or biased anchor;
- `mean` can suppress a valid mode when one alternative anchor is poor;
- a reliability-weighted cost quantile provides a train-selectable compromise.

For candidate cost values `c_j` and non-negative anchor reliabilities `w_j`, the
new selector sorts the finite costs and returns the first cost whose cumulative
normalized reliability reaches `q`.

```text
q = 0.0   -> weighted minimum
q = 0.5   -> weighted median
q = 1.0   -> weighted maximum
```

The quantile changes only the finite physical-hypothesis selection unary. The
final grouped mixture-MAP objective still uses the maintained candidate score,
learned sigma, physical-group multiplicity correction, Huber loss, and temporal
smoothness. Ground truth is not consumed at inference.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_weighted_anchor_quantile.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv robust=/path/to/robust_estimates.csv \
  --anchor-csv score_top1=/path/to/score_top1_estimates.csv \
  --anchor-csv frame_median=/path/to/frame_median_estimates.csv \
  --anchor-reliability robust=4 \
  --anchor-reliability score_top1=2 \
  --anchor-reliability frame_median=1 \
  --anchor-cost-quantile 0.5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_weighted_anchor_quantile
```

## Train-CV ablation

Select all settings using train folds before one public-validation run.

```text
anchor sets:
  robust + score_top1
  robust + score_top1 + frame_median

relative reliabilities:
  1:1
  2:1
  4:2:1
  8:2:1

anchor_cost_quantile:
  0.0, 0.25, 0.5, 0.75, 1.0

anchor_selection_weight:
  0.25, 0.5, 1.0, 2.0
```

Report pose MSE, P95, physical-group top-K oracle recall, selected group count,
anchor-cost disagreement, and runtime. A useful outcome is improved assignment
without increasing the final candidate budget or changing the robust smoother.
