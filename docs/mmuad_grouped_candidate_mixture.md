# MMUAD hypothesis-group-aware candidate mixture

Raw and train-calibrated MMUAD branches can contain multiple coordinate
hypotheses for the same physical point-cloud cluster. A flat candidate softmax
counts every row independently, so a cluster represented by both a raw and a
calibrated row receives more prior mass than a cluster represented once.

`raft-uav-mmuad-grouped-candidate-mixture-map` adds an opt-in multiplicity
correction before running the existing robust candidate-mixture MAP smoother.
Rows are grouped by `mmuad_calibration_origin_row` by default. For a group with
`n` sibling hypotheses, the wrapper subtracts

```text
correction_strength * log(n)
```

from each candidate log weight. With correction strength `1.0`, group evidence
is based on the mean rather than the sum of duplicated branch evidence. Raw and
calibrated siblings still compete normally within the group.

The default core `raft-uav-mmuad-candidate-mixture-map` behavior is unchanged.

## Example

```bash
raft-uav-mmuad-grouped-candidate-mixture-map \
  --candidates-csv outputs/branch_reservoir_candidates.csv \
  --output-dir outputs/grouped_mixture \
  --top-k 20 \
  --score-column branch_consensus_rank_score \
  --sigma-column predicted_sigma_m \
  --temperature 128 \
  --loss huber \
  --huber-delta 1 \
  --smoothness-weight 7200 \
  --hypothesis-group-correction-strength 1.0
```

Useful outputs include:

- `mmuad_group_corrected_candidates.csv`
- `mmuad_candidate_mixture_assignments.csv`
- `mmuad_candidate_mixture_estimates.csv`
- `mmuad_hypothesis_group_summary.json`

The assignment table includes per-candidate group IDs, group sizes, and final
group responsibility mass. Missing group IDs are treated as unique candidates
by default. Use `--missing-hypothesis-group-policy error` for strict pipelines.

This correction is inference-safe: it uses candidate metadata only and does not
require validation or test truth.
