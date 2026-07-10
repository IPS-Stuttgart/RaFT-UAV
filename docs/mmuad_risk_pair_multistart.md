# MMUAD risk-reservoir pair-prior multi-start pipeline

This experiment composes three inference-safe candidate-assignment improvements:

1. **Risk-adjusted branch-preserving reservoir**: retain candidates using calibrated score and train-predicted geometric uncertainty.
2. **Pair-state forward-backward prior**: attach soft candidate posteriors using adjacent motion and irregular-time constant-velocity acceleration factors.
3. **Branch-seeded multi-start Huber mixture-MAP**: solve the robust trajectory objective from global, median, and branch-specific initial trajectories and select the lowest truth-free objective.

The stages intentionally use uncertainty differently:

- reservoir selection uses the risk-adjusted score;
- the pair prior uses the original calibrated/ranker score and learned sigma;
- mixture-MAP uses the pair posterior as its score and learned sigma for residual scaling and measurement precision;
- the default mixture log-sigma prior is zero to avoid counting the pair prior's unary uncertainty term twice.

Validation/test truth is never used for reservoir selection, pair inference, restart construction, or restart selection. A truth CSV only enables scorecard and reservoir-oracle diagnostics.

## Example

```bash
python scripts/mmuad_candidate_risk_pair_multistart.py \
  --candidate-csv raw=/path/to/raw_candidates.csv \
  --candidate-csv dynamic=/path/to/dynamic_candidates.csv \
  --candidate-csv translated=/path/to/source_translated_candidates.csv \
  --output-dir outputs/mmuad_risk_pair_multistart \
  --truth-csv /path/to/public_validation_truth.csv \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --uncertainty-weight 1.0 \
  --transition-speed-std-mps 15 \
  --acceleration-std-mps2 20 \
  --max-acceleration-mps2 80 \
  --huber-delta 1.0 \
  --smoothness-weight 7200 \
  --iterations 5
```

## Main outputs

- `mmuad_risk_pair_multistart_scored_candidates.csv`
- `mmuad_risk_pair_multistart_reservoir.csv`
- `mmuad_risk_pair_multistart_pair_candidates.csv`
- standard selected mixture estimates, assignments, and iteration rows
- multi-start objective and initialization tables
- optional reservoir top-K oracle tables
- `mmuad_risk_pair_multistart_summary.json`

## Recommended ablation

Use train-selected parameters and compare:

1. risk reservoir + single-start mixture;
2. risk reservoir + multi-start mixture;
3. risk reservoir + pair prior + single-start mixture;
4. risk reservoir + pair prior + multi-start mixture.

This separates gains from top-K retention, second-order temporal assignment, and local initialization. The frozen public-validation run should be performed only after train-side selection.
