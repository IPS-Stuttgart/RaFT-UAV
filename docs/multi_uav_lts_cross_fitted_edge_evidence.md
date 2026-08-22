# Cross-fitted learned-edge evidence for Multi-UAV LTS

This experiment evaluates the advanced proposal-graph association components without
training on the sequences they score. It is the evidence gate for learned edge
likelihoods, swarm-relative association, similarity motion, and ambiguity-beam
reranking.

## Why this experiment exists

The permissive proposal bank has shown very high diagnostic recall, while a
single-hypothesis, no-birth decoder remained below the raw tracker. That makes
association the measured bottleneck. The repository already contains the relevant
advanced components, but fitting an edge model on all training sequences and then
scoring those same sequences would overstate its value.

The cross-fitted runner therefore uses the exact five scenario-stratified folds and
seed used by the guarded tournament. For fold `k`, it fits the edge model on the
other four folds and applies that model only to fold `k`. It then assembles the five
held-out outputs into one complete 102-sequence prediction directory. Consequently,
both pooled metrics and the tournament's fold metrics remain out of sample with
respect to edge-model fitting.

## Candidate ladder

The focused tournament compares the raw BoT-SORT control with six graph candidates:

1. `graph_delayed_translation` — delayed path cover plus translation common motion.
2. `graph_delayed_similarity` — the same decoder with guarded similarity motion.
3. `graph_delayed_swarm` — translation motion plus a hand-weighted local swarm cost.
4. `graph_edge_oof` — a cross-fitted calibrated same-identity edge model.
5. `graph_edge_swarm_oof` — the learned model plus the explicit swarm cost.
6. `graph_edge_swarm_beam_oof` — learned and swarm costs with ambiguity-component
   beam reranking.

This is intentionally a small hypothesis ladder rather than a broad parameter sweep.
The raw prediction set remains the deterministic fallback.

## Leakage and provenance contract

For every fold, the runner records and validates:

- the exact training and held-out sequence lists;
- disjointness of those lists and complete one-time held-out coverage;
- the selected sequences stored in both the fitted model and fit summary;
- the model digest used for each held-out prediction group;
- native image resolution for every generated sequence;
- exact content digests for every assembled candidate directory.

The proposal tracker never receives truth. Truth is consumed only by
`proposal_edge_model` to label edges in the complementary training folds. A mismatch
between a model's recorded panel and its assigned fold fails before scoring.

## Running the experiment

The dedicated workflow is
`.github/workflows/multi-uav-lts-cross-fitted-edge-evidence.yml`. It reuses the same
persistent detector/proposal cache as the existing public-evidence workflow, then
runs:

```bash
python scripts/run_multi_uav_lts_cross_fitted_edge_evidence.py \
  --inputs-json /path/to/public-inputs.json \
  --run-dir /path/to/evidence-run \
  --device 0 \
  --expected-sequence-count 102 \
  --expected-frame-count 77293 \
  --require-improvement
```

The workflow publishes compact fit summaries, model digests, per-candidate metrics,
and the guarded tournament result. Prediction payloads remain on the self-hosted
runner and are not uploaded.

## Acceptance boundary

A graph candidate is eligible only when it clears the repository's existing gates:

- mean five-fold Codabench HOTA gain of at least `0.001`;
- paired-bootstrap HOTA lower bound at least zero;
- mean Codabench MOTA and IDF1 drops no worse than `0.002`;
- worst-scenario HOTA drop no worse than `0.01`;
- exact complete sequence coverage and valid provenance.

If no candidate clears every gate, `raw` is the scientific result and no learned
candidate should be submitted.

## Test-time materialization after a positive result

Cross-fitted outputs are evaluation artifacts, not a hidden-test submission. After a
candidate wins, fit one edge model on all 102 training sequences using the same fixed
hyperparameters, apply the winning decoder to the test proposal bank, and package
that complete prediction set with the existing strict submission validator. Do not
mix the five fold-specific models into a test submission.
