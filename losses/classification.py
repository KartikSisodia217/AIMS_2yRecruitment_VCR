"""Standard cross-entropy loss for classification tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyScorer(nn.Module):
    """Standard cross-entropy loss for candidate selection.
    
    Used as baseline loss for both answer and rationale selection.
    Takes scores for N candidates and a label, computes CE loss.
    """
    
    def forward(self, scores: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            scores: Tensor of shape [batch_size, num_candidates] or [num_candidates]
            label: Tensor of shape [batch_size] or scalar
        
        Returns:
            Scalar loss tensor
        """
        # Handle both batched and unbatched inputs
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        return F.cross_entropy(scores, label)
