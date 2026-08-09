"""Test similarity functions."""
import pytest
import torch
from models.similarity import CosineSimilarity

def test_cosine_similarity_output_shape(cosine_similarity):
    query = torch.randn(128)
    candidates = torch.randn(4, 128)
    scores = cosine_similarity.compute(query, candidates)
    assert scores.shape == (4,)

def test_cosine_similarity_identical():
    sim = CosineSimilarity(temperature=1.0)
    query = torch.randn(128)
    candidates = torch.stack([query, query])
    scores = sim.compute(query, candidates)
    assert torch.allclose(scores, torch.ones_like(scores), atol=1e-5)

def test_cosine_similarity_orthogonal():
    sim = CosineSimilarity(temperature=1.0)
    query = torch.tensor([1.0, 0.0])
    candidates = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    scores = sim.compute(query, candidates)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-5)

def test_cosine_similarity_opposite():
    sim = CosineSimilarity(temperature=1.0)
    query = torch.tensor([1.0, 1.0])
    candidates = torch.tensor([[-1.0, -1.0]])
    scores = sim.compute(query, candidates)
    assert torch.allclose(scores, -torch.ones_like(scores), atol=1e-5)

def test_cosine_similarity_temperature():
    sim_high = CosineSimilarity(temperature=10.0)
    sim_low = CosineSimilarity(temperature=0.1)
    
    query = torch.randn(128)
    candidates = torch.randn(4, 128)
    
    scores_high = sim_high.compute(query, candidates)
    scores_low = sim_low.compute(query, candidates)
    
    # Lower temperature -> higher magnitude of scores
    assert torch.mean(torch.abs(scores_low)) > torch.mean(torch.abs(scores_high))

def test_dot_product_similarity_basic():
    # Placeholder for dot product similarity test if implemented
    pass
