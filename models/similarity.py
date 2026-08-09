"""
Similarity functions for comparing embeddings.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class SimilarityFunction(ABC):
    """Abstract similarity function between query and candidate embeddings."""
    
    @abstractmethod
    def compute(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and each candidate.
        
        Args:
            query: Tensor of shape [embedding_dim]
            candidates: Tensor of shape [N, embedding_dim]
        
        Returns:
            Tensor of shape [N] with similarity scores
        """
        ...


class CosineSimilarity(SimilarityFunction):
    """Cosine similarity between query and candidates."""
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # Normalize
        query_norm = F.normalize(query, p=2, dim=-1)
        cand_norm = F.normalize(candidates, p=2, dim=-1)
        # Cosine sim
        sims = torch.matmul(cand_norm, query_norm)
        return sims / self.temperature


class DotProductSimilarity(SimilarityFunction):
    """Simple dot product similarity (for comparison/ablation)."""
    
    def compute(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return torch.matmul(candidates, query)
