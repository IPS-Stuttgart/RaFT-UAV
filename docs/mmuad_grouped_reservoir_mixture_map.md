# Group-corrected branch-preserving MMUAD mixture MAP

The branch-preserving reservoir and origin-group correction solve different
failure modes:

1. the reservoir prevents low-ranked raw, calibrated, dynamic, or source-specific
   hypotheses from being pruned before trajectory inference;
2. the group correction prevents one physical observation from receiving extra
   prior mass merely because it is represented by several coordinate hypotheses.

Running only the reservoir can therefore create a representation-count bias,
especially for calibration-path ensembles. Running only grouped mixture-MAP can
still lose useful branches through its global top-K truncation. The composed
runner applies the reservoir first, forces downstream `top_k=0`, and then applies
`log(group_size)` correction to the retained sibling representations.

## Run

```bash
python scripts/mmuad_grouped_reservoir_mixture_map.py \
  --candidate-csv raw=outputs/mmuad/raw_candidates.csv \
  --candidate-csv calibrated=outputs/mmuad/calibrated_candidates.csv \
  --output-dir outputs/mmuad/grouped_reservoir_map \
  --global-top-n 20 \
  --per-source-top-n 3 \
  --per-branch-top-n 3 \
  --max-candidates-per-frame 40 \
  --hypothesis-group-correction-strength 1
```

The group column is auto-detected from `mmuad_calibration_origin_row`,
`candidate_origin_row`, or `origin_row`. Use `--hypothesis-group-column` to
override it. Rows without a group are unique by default; use
`--missing-hypothesis-group-policy error` for strict provenance checks.

## Train-CV ablation

Freeze all choices on training folds and compare:

```text
group correction: 0, 0.5, 0.75, 1
reservoir branch quota: 1, 2, 3
reservoir cap: 20, 40, 60
```

Report pose MSE/P95 together with retained-reservoir oracle recall, duplicate
hypothesis group counts, effective hypothesis group counts, candidate count, and
runtime. The correction is inference-safe and does not use validation/test truth.
