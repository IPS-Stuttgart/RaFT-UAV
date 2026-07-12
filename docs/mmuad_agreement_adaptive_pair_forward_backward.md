# Agreement-adaptive pair forward-backward prior

The acceleration-aware pair-state posterior can recover coherent candidates that
are buried by the framewise ranker. Entropy-adaptive blending already backs off
when that posterior is diffuse, but a wrong smooth mode can still be sharply
concentrated and therefore appear confident.

This variant also measures disagreement between the local emission posterior
and pair-state posterior with normalized Jensen-Shannon divergence:

```text
confidence = 1 - H(pair) / log(K)
agreement  = 1 - JS(local, pair) / log(2)
trust      = confidence^p * agreement^q
lambda     = lambda_min + (lambda_max - lambda_min) * trust
p_out      proportional to local^(1-lambda) * pair^lambda
```

A decisive pair posterior is trusted when it remains compatible with the local
learned score/uncertainty model. A decisive but contradictory posterior backs
off toward the local model instead of committing to a potentially smooth
clutter trajectory. Setting `agreement_power=0` exactly recovers the existing
entropy-only adaptive blend.

## Run

```bash
python scripts/mmuad_candidate_pair_forward_backward_agreement.py \
  --candidate-csv outputs/mmuad_reservoir/candidates.csv \
  --output-csv outputs/mmuad_pair_agreement/candidates.csv \
  --summary-json outputs/mmuad_pair_agreement/summary.json \
  --score-column candidate_reservoir_grid_score \
  --sigma-column predicted_sigma_m \
  --min-pair-weight 0.0 \
  --max-pair-weight 1.0 \
  --confidence-power 1.0 \
  --agreement-power 1.0 \
  --mixture-output-dir outputs/mmuad_pair_agreement/mixture \
  --mixture-top-k 20 \
  --mixture-smoothness-weight 7200 \
  --mixture-huber-delta 1
```

Ground truth is never used to construct the candidate prior. Optional mixture
truth is passed only to the standard score reporter.

## Train-CV ablation

Freeze the candidate pool, local score model, learned sigma, pair transition
model, Huber loss, and smoother. Select only:

```text
min_pair_weight:   0, 0.1, 0.25
max_pair_weight:   0.5, 0.75, 1
confidence_power:  0.5, 1, 2, 4
agreement_power:   0, 0.5, 1, 2, 4
```

Compare pose MSE/P95, top-candidate changes, normalized JS divergence,
effective pair weight, and failure sequences. The `agreement_power=0` row is
the entropy-only baseline. A useful outcome reduces catastrophic smooth-mode
errors without losing the deeper-candidate recoveries of pair-state inference.
