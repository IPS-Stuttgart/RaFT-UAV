import numpy as np
from raft_uav.multi_uav_lts.observable_expert_gate import fit_gate

def test_gate_selects_only_confident_gain():
    features = {'a': np.zeros(10), 'b': np.ones(10), 'c': np.ones(10) * 2, 'd': np.ones(10) * 3}
    scores = {name: {'raw': 0.5, 'expert': 0.5 + 0.01 * i} for i, name in enumerate(features)}
    model = fit_gate(features, scores, l2=0.01, margin=0.0001)
    assert model.choose(np.ones(10) * 3) == 'expert'
    conservative = fit_gate(features, {k: {'raw': 0.5, 'expert': 0.5} for k in features}, margin=0.0001)
    assert conservative.choose(np.zeros(10)) == 'raw'
