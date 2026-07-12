# Agreement-adaptive pair forward-backward prior

The pair-state forward-backward model can recover temporally coherent MMUAD
candidates that are buried by a local ranker. Entropy-adaptive blending already
reduces the influence of an ambiguous pair posterior, but entropy alone cannot
detect a sharp posterior that confidently follows the wrong smooth mode.

This variant additionally measures agreement between the local emission
posterior and the pair-state posterior using normalized Jensen-Shannon
divergence:

```text
agreement = 1 - JS(local, pair) / log(2)
```

The effective pair weight is

```text
pair_confidence = 1 - normalized_pair_entropy
agreement_factor = floor + (1 - floor) * agreement ** agreement_power

lambda = lambda_min
         + (lambda_max - lambda_min)
           * pair_confidence ** entropy_power
           * agreement_factor
```

The final candidate posterior is the normalized geometric blend

```text
p(c) proportional to p_local(c) ** (1 - lambda) * p_pair(c) ** lambda.
```

A confident pair posterior therefore receives high weight only when it is also
reasonably compatible with the local learned score/uncertainty evidence. No
ground truth is consumed during inference.

## Example

```bash
python scripts/mmuad_candidate_pair_forward_backward_agreement_adaptive.py \
  --candidate-csv outputs/train_selected_reservoir.csv \
  --output-csv outputs/pair_agreement_candidates.csv \
  --summary-json outputs/pair_agreement_summary.json \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --min-pair-weight 0 \
  --max-pair-weight 1 \
  --entropy-power 1 \
  --agreement-power 2 \
  --agreement-floor 0.1
```

The command can also hand the resulting posterior directly to the existing
learned-sigma Huber mixture-MAP path through `--mixture-output-dir`.

## Train-only ablation

Select all controls on train folds and freeze them before public validation or
hidden-test inference.

```text
entropy_power:  0.5, 1, 2, 4
agreement_power: 0.5, 1, 2, 4
agreement_floor: 0, 0.05, 0.1, 0.25, 0.5
min_pair_weight: 0, 0.1, 0.25
max_pair_weight: 0.5, 0.75, 1
```

Compare:

- entropy-only adaptive pair prior;
- agreement-adaptive pair prior;
- local score/uncertainty posterior;
- raw pair-state posterior.

Report pose MSE, P95, top-K oracle recall, mean effective pair weight,
Jensen-Shannon divergence, and the fraction of frames where the adaptive top
candidate differs from each expert.
