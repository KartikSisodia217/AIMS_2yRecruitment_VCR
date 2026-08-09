"""Test projection head."""
import pytest
import torch
from models.projection import ProjectionHead

def test_projection_output_shape():
    head = ProjectionHead(input_dim=256, embedding_dim=128)
    x = torch.randn(256)
    out = head(x)
    assert out.shape == (128,)

def test_projection_batch_output_shape():
    head = ProjectionHead(input_dim=256, embedding_dim=128)
    x = torch.randn(4, 256)
    out = head(x)
    assert out.shape == (4, 128)

def test_projection_l2_normalized():
    head = ProjectionHead(input_dim=256, embedding_dim=128, normalize=True)
    x = torch.randn(4, 256)
    out = head(x)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))

def test_projection_not_l2_normalized():
    head = ProjectionHead(input_dim=256, embedding_dim=128, normalize=False)
    x = torch.randn(4, 256)
    out = head(x)
    norms = torch.norm(out, p=2, dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms))

def test_projection_gradient_flows():
    head = ProjectionHead(input_dim=256, embedding_dim=128)
    x = torch.randn(4, 256, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_projection_activations():
    x = torch.randn(4, 256)
    for act in ["gelu", "relu", "tanh", "silu"]:
        head = ProjectionHead(input_dim=256, embedding_dim=128, activation=act)
        out = head(x)
        assert out.shape == (4, 128)

def test_projection_dropout():
    head = ProjectionHead(input_dim=256, embedding_dim=128, dropout=0.5)
    head.train()
    x = torch.ones(4, 256)
    out1 = head(x)
    out2 = head(x)
    # With dropout 0.5, outputs should differ in training mode
    assert not torch.allclose(out1, out2)
    
    head.eval()
    out3 = head(x)
    out4 = head(x)
    # In eval mode, outputs should be deterministic
    assert torch.allclose(out3, out4)
