# Confidence-adaptive MMUAD class conditioning

The fixed-strength class-conditioned anchor selector assumes that the UAV type
posterior is equally trustworthy for every sequence. That is unnecessarily
aggressive for ambiguous or near-uniform class probabilities.

This experiment scales class conditioning per candidate row:

```text
lambda_eff = lambda_max * (floor + (1 - floor) * confidence(p)^power)
```

The effective anchor reliability is then

```text
w_eff(anchor) = (1 - lambda_eff) * w_global(anchor)
                + lambda_eff * sum_c p(c) * w(anchor, c)
```

A confident posterior therefore uses the train-selected class-specific anchor
profile. A uniform posterior backs off to the global anchor profile. The final
grouped learned-sigma / Huber mixture-MAP objective is unchanged.

## Confidence modes

- `entropy`: one minus normalized categorical entropy;
- `max-probability`: maximum probability above the uniform baseline;
- `margin`: largest minus second-largest probability;
- `none`: reproduces fixed-strength conditioning.

## Example

```bash
python scripts/mmuad_candidate_mixture_group_confidence_adaptive_class_anchor_quantile.py \
  --candidates-csv candidates.csv \
  --anchor-csv robust=robust_estimates.csv \
  --anchor-csv score_top1=score_top1_estimates.csv \
  --class-probabilities-csv class_probabilities.csv \
  --anchor-reliability robust=4 \
  --anchor-reliability score_top1=1 \
  --anchor-class-reliability robust:0=8 \
  --anchor-class-reliability score_top1:0=1 \
  --class-conditioning-strength 1 \
  --class-confidence-mode entropy \
  --class-confidence-power 1 \
  --class-confidence-floor 0 \
  --anchor-cost-quantile 0.5 \
  --output-dir outputs/mmuad_confidence_adaptive_class
```

## Train-CV ablation

Select all settings on train folds using out-of-fold class probabilities:

```text
confidence_mode:          entropy, max-probability, margin, none
confidence_power:         0.5, 1, 2, 4
confidence_floor:         0, 0.1, 0.25, 0.5
conditioning_strength:    0.25, 0.5, 0.75, 1
anchor_cost_quantile:     0.25, 0.5, 0.75
```

Report pooled and per-class pose MSE, P95, selected physical-group count,
confidence distribution, effective conditioning strength, anchor disagreement,
and runtime. Ground truth is never used for inference-time weighting or
selection.
