# Learned swarm-aware edge likelihood for Multi-UAV LTS

## Scientific hypothesis

The permissive 1920-pixel proposal bank already has near-oracle recall on the
existing pilot, while the realized tracker remains much weaker. The next
association milestone therefore replaces hand-weighted edge ordering with a
calibrated probability that two proposals belong to the same physical UAV.

The model is intentionally small. It is a regularized logistic classifier over
features that remain available on the hidden test set:

- common-motion-compensated center residual;
- log width/height change;
- predicted-box IoU loss;
- proposal confidence deficit;
- gap length;
- local swarm-relative geometry error;
- swarm-feature support; and
- local-density change.

The swarm term compares small permutation-tolerant constellations of normalized
neighbor vectors around the two candidate proposals. It can disambiguate edges
that have similar unary motion cost but imply different local formation
geometry.

## Fit a model on training sequences

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.proposal_edge_model \
  /path/to/train/proposals \
  --truth-dir /path/to/TrainLabels \
  --output-json /path/to/edge-model.json \
  --summary-json /path/to/edge-model-summary.json \
  --examples-csv /path/to/edge-examples.csv \
  --enable-common-motion \
  --min-truth-iou 0.3 \
  --negative-candidates-per-left 5 \
  --swarm-neighbors 4 \
  --swarm-radius-scale 12 \
  --l2-penalty 1
```

Truth is used only to label training edges. Proposal-to-truth matching is
one-to-one within each frame. Every valid positive continuation is retained;
only the lowest-cost hard negatives per left proposal are used, preventing the
fit from being dominated by trivial distant negatives. Training weights are
balanced by physical sequence and by class.

The JSON model records the exact feature schema, standardization statistics,
coefficients, class counts, sequence count, fit controls, and selected sequence
names. Runtime loading rejects incompatible or malformed models.

## Apply the frozen model

```bash
PYTHONPATH=src RAFT_UAV_SKIP_RUNTIME_HOOKS=1 \
python -m raft_uav.multi_uav_lts.experimental_proposal_graph_tracker \
  /path/to/test/proposals \
  --first-frame-label-dir /path/to/TestLabels_FirstFrameOnly \
  --output-dir /path/to/test_learned_edges/predictions \
  --output-json /path/to/test_learned_edges/summary.json \
  --enable-delayed-path-cover \
  --delayed-max-gap 0 \
  --delayed-lookahead-frames 2 \
  --delayed-successors-per-frame 3 \
  --delayed-continuation-weight 0.75 \
  --enable-common-motion \
  --edge-model-json /path/to/edge-model.json \
  --edge-model-weight 1 \
  --edge-model-clip 4 \
  --swarm-neighbors 4 \
  --swarm-radius-scale 12
```

The model contributes only a **relative** bounded penalty among candidates for
the same source proposal and target frame. The locally best learned edge keeps
its original geometric cost. Consequently, the model can reorder ambiguous
candidates without globally making every link easier or manufacturing extra
links outside the existing geometric gate.

A truth-free direct swarm ablation is also available:

```text
--swarm-relative-weight 0.75
--swarm-relative-clip 4
--swarm-neighbors 4
--swarm-radius-scale 12
```

This should be evaluated before the learned model. It tests whether the swarm
feature itself has signal independently of fitting. The existing public-evidence
runner can include this candidate without changing the detector cache:

```bash
python scripts/run_multi_uav_lts_swarm_edge_evidence.py \
  --inputs-json /path/to/public-inputs.json \
  --run-dir /path/to/run \
  --device 0 \
  --expected-sequence-count 102 \
  --expected-frame-count 77293 \
  --require-improvement
```

## Leakage-safe evaluation

Do not fit one model on all 102 training sequences and then report its score on
those same sequences. Use the existing deterministic scenario-stratified folds:

1. fit the model on four folds using `--sequences <train names>`;
2. run the tracker only on the held-out names using `--sequences <held-out names>`;
3. combine the five disjoint held-out prediction directories;
4. compare the complete out-of-fold candidate against the exact raw control in
   `raft_uav.multi_uav_lts.tournament`; and
5. only after the method passes, refit once on all training sequences for the
   frozen hidden-test submission.

The acceptance gate remains unchanged: positive held-out `CODABENCH_HOTA`, a
positive paired-bootstrap lower bound, acceptable MOTA and IDF1 changes, no
excessive worst-prefix regression, complete sequence coverage, and exact
provenance.

The expected successful signature is primarily higher AssA/IDF1 and fewer ID
switches. A gain caused mainly by extra output rows is not evidence that the
association model solved the intended problem.
