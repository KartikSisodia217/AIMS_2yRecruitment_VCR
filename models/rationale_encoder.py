"""
Rationale Encoder models for Stage 2.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
from .vlm_backbone import VLMBackbone
from .projection import ProjectionHead


class RationaleEncoder(ABC):
    """Abstract interface for encoding rationale candidates.
    
    Produces embeddings in the same space as the context embedding
    (output of ProjectionHead) for similarity comparison.
    
    # RESEARCH_DECISION: Should the encoder share weights with the VLM?
    # RESEARCH_DECISION: Should rationales be encoded with or without context?
    # RESEARCH_DECISION: What text representation works best?
    """
    
    @abstractmethod
    def encode(self, rationales: List[str], 
              context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Encode rationale candidates.
        
        Args:
            rationales: List of N rationale texts
            context: Optional context (question, answer, image info)
        
        Returns:
            Tensor of shape [N, embedding_dim]
        """
        ...
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...


class MockRationaleEncoder(RationaleEncoder):
    """Mock encoder for testing. Returns deterministic embeddings."""
    
    def __init__(self, embedding_dim: int = 512):
        self._embedding_dim = embedding_dim
    
    def encode(self, rationales: List[str], context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        embeddings = []
        for r in rationales:
            seed = hash(r) % (2**31)
            gen = torch.Generator().manual_seed(seed)
            emb = torch.randn(self._embedding_dim, generator=gen)
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            embeddings.append(emb)
        return torch.stack(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class ProjectionRationaleEncoder(RationaleEncoder, nn.Module):
    """Encodes rationales using VLM + projection head.
    
    For each rationale, runs it through the VLM's text encoder (optionally
    with context) and projects to the embedding space.
    
    # RESEARCH_DECISION: This is one possible approach. Alternatives include
    # using a separate text encoder or sharing the context projection head.
    """
    
    def __init__(self, vlm: VLMBackbone, projection: ProjectionHead):
        super().__init__()
        self.vlm = vlm
        self.projection = projection
    
    def encode(self, rationales: List[str], context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        embeddings = []
        for r in rationales:
            # Get VLM text representation (no image)
            hidden = self.vlm.encode_text_only(r)
            # Project to embedding space
            emb = self.projection(hidden.unsqueeze(0)).squeeze(0)
            embeddings.append(emb)
        return torch.stack(embeddings)
    
    @property
    def embedding_dim(self) -> int:
        return self.projection.embedding_dim  # expose from projection head
