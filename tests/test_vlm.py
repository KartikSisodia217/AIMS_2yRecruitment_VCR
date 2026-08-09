"""Test VLM backbone."""
import pytest
import torch

def test_mock_vlm_hidden_dim(mock_vlm):
    assert mock_vlm.get_hidden_dim() == 256

def test_mock_vlm_encode_shape(mock_vlm):
    out = mock_vlm.encode(None, "hello")
    assert out.shape == (256,)

def test_mock_vlm_encode_text_only_shape(mock_vlm):
    out = mock_vlm.encode_text_only("hello")
    assert out.shape == (256,)

def test_mock_vlm_deterministic(mock_vlm):
    out1 = mock_vlm.encode(None, "hello")
    out2 = mock_vlm.encode(None, "hello")
    assert torch.allclose(out1, out2)

def test_mock_vlm_differs_for_different_inputs(mock_vlm):
    out1 = mock_vlm.encode(None, "hello")
    out2 = mock_vlm.encode(None, "world")
    assert not torch.allclose(out1, out2)

def test_mock_vlm_encode_vs_encode_text_only(mock_vlm):
    out1 = mock_vlm.encode(None, "hello")
    out2 = mock_vlm.encode_text_only("hello")
    # Might differ depending on implementation, prompt asked to test it differs
    assert not torch.allclose(out1, out2)

def test_mock_vlm_compute_log_likelihood(mock_vlm):
    ll = mock_vlm.compute_log_likelihood(None, "prompt", "completion")
    assert isinstance(ll, float)

def test_mock_vlm_freeze_unfreeze(mock_vlm):
    mock_vlm.freeze()
    for p in mock_vlm.get_trainable_parameters():
        assert not p.requires_grad
    mock_vlm.unfreeze()
    for p in mock_vlm.get_trainable_parameters():
        assert p.requires_grad

def test_mock_vlm_get_total_params(mock_vlm):
    assert mock_vlm.get_total_params() > 0

def test_mock_vlm_device(mock_vlm):
    assert mock_vlm.device == torch.device('cpu') or mock_vlm.device.type == 'cpu'
