"""
Projection Head for CACR-SP.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projects VLM hidden representations into the CACR embedding space.
    
    This is OUR task-specific component. It transforms the VLM's general-purpose
    representation into an embedding space suitable for contrastive comparison
    with rationale embeddings.
    
    The VLM does the visual understanding.
    This projection head transforms the resulting representation.
    
    Architecture:
        VLM hidden state [hidden_dim]
        → Linear(hidden_dim, intermediate_dim)
        → Activation
        → Dropout
        → Linear(intermediate_dim, embedding_dim)
        → Optional L2 normalization
    
    # RESEARCH_DECISION: optimal architecture, dimensions, normalization
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 512, 
                 intermediate_dim: int = 1024, activation: str = "gelu",
                 dropout: float = 0.1, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self._embedding_dim = embedding_dim
        
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, embedding_dim)
        )
    
    @property
    def embedding_dim(self) -> int:
        """Returns the output embedding dimension."""
        return self._embedding_dim

    @classmethod
    def from_config(cls, config: dict, input_dim: int) -> "ProjectionHead":
        """Instantiate from a configuration dictionary."""
        return cls(
            input_dim=input_dim,
            embedding_dim=config.get("embedding_dim", 512),
            intermediate_dim=config.get("intermediate_dim", 1024),
            activation=config.get("activation", "gelu"),
            dropout=config.get("dropout", 0.1),
            normalize=config.get("normalize", True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden representation to embedding space.
        
        Args:
            x: Tensor of shape [..., input_dim]
        
        Returns:
            Tensor of shape [..., embedding_dim], optionally L2-normalized
        """
        out = self.net(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
