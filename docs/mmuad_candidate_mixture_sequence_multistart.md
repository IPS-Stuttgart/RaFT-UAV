# MMUAD per-sequence candidate-mixture multi-start

`candidate_mixture_map_multistart` evaluates several initial trajectories but
selects one restart for the whole input batch.  That is unnecessarily
restrictive for MMUAD because each sequence has its own independent trajectory
objective and different sequences may prefer different candidate branches.

The per-sequence runner evaluates the same truth-free robust objective for each
restart and sequence, chooses the best restart separately for every sequence,
and performs one final candidate-mixture run using the combined initialization.
Validation/test truth is never used for restart selection.

## Run

```bash
python -m raft_uav.mmuad.candidate_mixture_map_sequence_multistart \
  --candidates-csv outputs/risk_pair/mmuad_risk_pair_multistart_pair_candidates.csv \
  --output-dir outputs/sequence_multistart \
  --top-k 0 \
  --score-column candidate_pair_fb_log_posterior \
  --score-normalization none \
  --sigma-column predicted_sigma_m_branch_class \
  --sigma-log-weight 0 \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200
```

For a local public-validation diagnostic, add `--truth-csv`.  Truth only adds
metrics to the individual smoother runs; the selected restart is still based
on the final robust mixture evidence, acceleration regularizer, and optional
initialization-anchor term.

## Outputs

The output directory contains the standard selected candidate-mixture files
plus:

- `mmuad_candidate_mixture_sequence_multistart_summary.csv`: one row per
  sequence and restart, including the truth-free selection objective and the
  selected flag;
- `mmuad_candidate_mixture_sequence_multistart_summary.json`: selected restart
  by sequence and configuration provenance;
- `mmuad_candidate_mixture_sequence_multistart_initializations.csv`: all
  explicit restart trajectories;
- `mmuad_candidate_mixture_sequence_multistart_selected_initializations.csv`:
  the sequence-specific initialization used for the final inference run.

## Intended ablation

Keep the risk reservoir, pair-state prior, learned sigma column, Huber loss, and
trajectory hyperparameters fixed.  Compare:

1. the existing batch-wide multi-start selection;
2. per-sequence restart selection;
3. the oracle best restart per sequence as a diagnostic ceiling only.

The per-sequence method is inference-safe and should never be selected using
validation/test pose error.  Its expected benefit is on heterogeneous
sequences where raw, translated, dynamic, or external starts converge to
different local mixture-MAP solutions.
