import numpy as np
import pytest
torch = pytest.importorskip('torch')
from raft_uav.multi_uav_lts.tiny_p2_detector import TinyP2Detector, _tensor_loss, size_adaptive_nwd_loss_np

def test_tiny_boxes_receive_more_nwd_weight():
    pred = np.array([[12.0, 12.0, 12.0, 12.0], [12.0, 12.0, 48.0, 48.0]])
    target = np.array([[14.0, 12.0, 12.0, 12.0], [14.0, 12.0, 48.0, 48.0]])
    loss = size_adaptive_nwd_loss_np(pred, target)
    assert loss[0] > loss[1]

def test_p2_head_and_loss_are_finite():
    model = TinyP2Detector(channels=16, blocks=1)
    output = model(torch.zeros(1, 1, 64, 64))
    assert output[0].shape[-2:] == (16, 16)
    loss, parts = _tensor_loss(output, [np.array([[20.0, 20.0, 8.0, 8.0]], dtype=np.float32)])
    assert torch.isfinite(loss)
    assert parts['nwd'] >= 0
