# MMUAD per-sequence candidate-pool selection

The branch-preserving MMUAD candidate reservoir avoids discarding useful raw,
dynamic, translated, calibrated, or merged candidates before trajectory
inference. A branch can still be useful for one sequence and harmful for
another, so one global branch ablation is unnecessarily restrictive.

`candidate_mixture_map_sequence_pool_selector` evaluates the full candidate
pool and leave-one-group-out variants with the same truth-free robust
candidate-mixture objective, then selects a pool independently for every
sequence.

## Method

For each eligible `candidate_branch` or `source`, the selector builds:

- the full candidate pool;
- a pool with that group removed;
- optional full-pool fallback rows for timestamps that would otherwise lose all
  candidates.

Every pool is passed through the same learned-sigma / Huber mixture-MAP
configuration. The restart objective is evaluated per sequence. Because a raw
`logsumexp` mixture score systematically rewards pools containing more
components, the selector compares a component-count-normalized objective:

```text
negative log mean mixture evidence
  = negative logsumexp evidence + sum_t log(K_t)
```

where `K_t` is the number of retained candidates at timestamp `t`. This keeps a
larger pool from winning solely because it contains more candidates.

Truth is optional and is used only for diagnostic metrics. It is never used to
construct pools or select the winning pool.

## Example

```bash
python scripts/mmuad_candidate_mixture_sequence_pool_selector.py \
  --candidates-csv outputs/full_branch_pool/candidates.csv \
  --truth-csv /mnt/lexar4tb/mmuad_realdata/challenge_meta/validation_ref_new_for_your_ref.csv \
  --output-dir outputs/mmuad_sequence_pool_selector \
  --group-column candidate_branch \
  --max-leave-one-out 8 \
  --min-group-frame-fraction 0.05 \
  --top-k 20 \
  --score-column candidate_pair_forward_backward_score \
  --sigma-column predicted_sigma_m_branch_class \
  --loss huber \
  --huber-delta 1.0 \
  --smoothness-weight 7200 \
  --iterations 5
```

The command writes the normal candidate-mixture estimate, assignment,
iteration, and summary artifacts plus:

```text
mmuad_candidate_mixture_sequence_pool_summary.csv
mmuad_candidate_mixture_sequence_pool_summary.json
mmuad_candidate_mixture_sequence_pool_candidates.csv
```

The CSV summary contains the truth-free objective for every sequence/pool pair,
the mixture-component correction, candidate coverage, optional diagnostic
metrics, and the selected pool.

## Recommended ablation

Keep the candidate reservoir, temporal prior, learned uncertainty, Huber loss,
smoothness, and initialization fixed. Compare:

1. the full pool only;
2. global leave-one-branch-out selection on train CV;
3. per-sequence leave-one-branch-out selection from this module.

The most relevant first check is whether sequences previously harmed by source
translation select `without_candidate_branch_source_translation`, while
sequences benefiting from calibration retain the full or translated pool.
