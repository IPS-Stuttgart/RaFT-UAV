# MMUAD candidate oracle attribution

`raft-uav-mmuad-candidate-oracle-attribution` explains which candidate source and
candidate branch supplies the nearest-to-truth candidate in a candidate pool or
branch-preserving reservoir. It is intended for public-validation diagnostics
and train-fold analyses before running expensive mixture-MAP sweeps.

Example:

```bash
raft-uav-mmuad-candidate-oracle-attribution \
  --candidate raw_static=/path/to/raw_static_candidates.csv \
  --candidate dynamic=/path/to/dynamic_candidates.csv \
  --candidate source_translation=/path/to/source_translated_candidates.csv \
  --truth-csv /path/to/reference.csv \
  --output-dir outputs/mmuad_candidate_oracle_attribution
```

The command writes:

- `mmuad_candidate_oracle_attribution_frames.csv`
- `mmuad_candidate_oracle_attribution_pooled.csv`
- `mmuad_candidate_oracle_attribution_by_branch.csv`
- `mmuad_candidate_oracle_attribution_by_source.csv`
- `mmuad_candidate_oracle_attribution_summary.json`

The key columns are `oracle_all_candidate_branch`,
`oracle_all_candidate_source`, and `oracle_all_rank`. A high oracle rank means
that good candidates are present but buried by the current score; a poor
all-candidate oracle indicates an extraction or calibration failure.
