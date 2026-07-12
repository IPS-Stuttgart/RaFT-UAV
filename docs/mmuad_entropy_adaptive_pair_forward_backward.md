# Entropy-adaptive pair forward-backward prior

The pair-state forward-backward model adds acceleration-aware trajectory evidence to
MMUAD candidate scores. It can recover a physically coherent candidate that is buried
by the per-frame ranker, but a globally smooth distractor can also make the pair-state
posterior overconfident.

This experiment keeps both signals and controls their influence frame by frame.
For local emission posterior `p_local`, pair-state posterior `p_pair`, and normalized
pair entropy `H_norm`, it uses

```text
confidence = 1 - H_norm
lambda = lambda_min + (lambda_max - lambda_min) * confidence ** power
p_adaptive ∝ p_local ** (1 - lambda) * p_pair ** lambda
```

Thus an ambiguous near-uniform pair posterior backs off toward the local learned
score/uncertainty model, while a decisive pair posterior retains its trajectory-level
influence. The adaptive posterior can be passed directly to the existing learned-sigma
Huber mixture-MAP smoother.

Inference uses candidate geometry, timestamps, source/branch/track metadata, candidate
scores, and learned uncertainty only. Truth is optional and is used only by the
mixture-MAP diagnostic path.

## Run

```bash
python scripts/mmuad_candidate_pair_forward_backward_adaptive.py \
  --candidate-csv /path/to/branch_preserving_candidates.csv \
  --output-csv outputs/mmuad_adaptive_pair/candidates.csv \
  --summary-json outputs/mmuad_adaptive_pair/summary.json \
  --score-column candidate_risk_adjusted_score \
  --sigma-column predicted_sigma_m \
  --min-pair-weight 0 \
  --max-pair-weight 1 \
  --confidence-power 1 \
  --mixture-output-dir outputs/mmuad_adaptive_pair/mixture \
  --mixture-truth-csv /path/to/train_or_public_validation_truth.csv \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1
```

## Train-only ablation

Select all settings on training folds before public-validation or hidden-test use.
A compact initial grid is:

```text
min_pair_weight:          0, 0.1, 0.25
max_pair_weight:          0.5, 0.75, 1
confidence_power:         0.5, 1, 2, 4
transition_distance_std:  1, 2, 5 m
transition_speed_std:     5, 15, 30 m/s
acceleration_std:         5, 10, 20, 40 m/s^2
```

Compare pose MSE, P95, top-K oracle recall after scoring, the fraction of frames whose
adaptive top candidate differs from the local and pair-state tops, effective pair
weight, normalized entropy, and runtime.

The intended decision rule is simple:

- keep the method only if train-CV selects a nontrivial adaptive range;
- reject it if the selector collapses to `lambda_min = lambda_max = 0`;
- use a fixed pair prior only if train-CV collapses to equal nonzero bounds.
