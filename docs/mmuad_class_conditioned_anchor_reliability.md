# MMUAD class-conditioned anchor reliability

The current MMUAD pose bottleneck is candidate assignment rather than the
maintained Huber smoother. Multi-anchor physical-group selection preserves
several inference-time trajectory modes, but fixed anchor reliability assumes
that every anchor has the same relative quality for every UAV type.

This experiment uses the fused sequence classifier as soft context for anchor
selection. For sequence class probabilities `p(c)`, global anchor reliability
`w_global(a)`, class-specific reliability `w(a,c)`, and blend `lambda`, the
inference-time weight is

```text
w_eff(a) = (1 - lambda) * w_global(a)
           + lambda * sum_c p(c) * w(a,c)
```

These weights are used only by the reliability-weighted anchor-cost quantile
that selects a bounded set of physical candidate groups. The final grouped
candidate-mixture MAP objective remains unchanged: learned candidate sigma,
physical-group multiplicity correction, Huber loss, and temporal smoothness.
No validation/test truth is consumed by selection.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_class_conditioned_anchor_quantile.py \
  --candidates-csv /path/to/reservoir_candidates.csv \
  --anchor-csv robust=/path/to/robust_estimates.csv \
  --anchor-csv score_top1=/path/to/score_top1_estimates.csv \
  --class-probabilities-csv /path/to/fused_class_probabilities.csv \
  --anchor-reliability robust=2 \
  --anchor-reliability score_top1=1 \
  --anchor-class-reliability robust:0=4 \
  --anchor-class-reliability score_top1:0=1 \
  --anchor-class-reliability robust:2=1 \
  --anchor-class-reliability score_top1:2=4 \
  --class-conditioning-strength 0.5 \
  --anchor-cost-quantile 0.5 \
  --anchor-selection-weight 0.5 \
  --anchor-scale-m 10 \
  --min-group-top-k 3 \
  --max-group-top-k 20 \
  --target-posterior-mass 0.95 \
  --hypothesis-group-column origin_row \
  --output-dir outputs/mmuad_class_conditioned_anchor_quantile
```

Unspecified anchor/class pairs fall back to the corresponding global anchor
reliability. Missing class probabilities default to a uniform distribution;
`--fill-missing-class-probabilities zero` falls back to global anchor weights
for missing sequences rather than producing an empty selector.

## Train-CV protocol

Estimate or select the class-by-anchor reliability matrix only from train folds.
Use out-of-fold class probabilities for train candidate rows, then fit the final
classifier on all training sequences before public validation or hidden test.

Suggested first grid:

```text
anchor sets:
  robust + score_top1
  robust + score_top1 + frame_median

class_conditioning_strength:
  0.0, 0.25, 0.5, 0.75, 1.0

anchor_cost_quantile:
  0.25, 0.5, 0.75

per-class reliability ratios:
  1:1
  2:1
  4:1
```

Report pooled and per-class pose MSE, P95, physical-group oracle recall,
selected-group count, effective anchor weights, classifier confidence, and
runtime. The main success condition is improved assignment at the same finite
candidate budget; classification accuracy itself is not the optimization
target.
