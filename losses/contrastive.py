"""Contrastive losses for rationale selection."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Contrastive ranking loss for CACR.
    
    Supports multiple loss types:
    - 'infonce': InfoNCE / softmax contrastive loss
    - 'margin_ranking': Margin-based ranking loss
    
    For InfoNCE with 4 candidates:
    L = -log(exp(sim(c, r+) / tau) / sum_i(exp(sim(c, ri) / tau)))
    
    This is equivalent to cross-entropy on the similarity scores.
    
    # RESEARCH_DECISION: optimal loss type and temperature
    """
    
    def __init__(self, loss_type: str = 'infonce', temperature: float = 0.07,
                 margin: float = 0.2):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, similarity_scores: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            similarity_scores: Shape [batch_size, num_candidates] or [num_candidates]
                Similarity between context embedding and rationale embeddings.
            label: Shape [batch_size] or scalar. Index of correct rationale.
        
        Returns:
            Scalar loss tensor
        """
        if self.loss_type == 'infonce':
            return self._infonce(similarity_scores, label)
        elif self.loss_type == 'margin_ranking':
            return self._margin_ranking(similarity_scores, label)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _infonce(self, scores, label):
        # scores / temperature -> cross_entropy
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        scaled = scores / self.temperature
        return F.cross_entropy(scaled, label)
    
    def _margin_ranking(self, scores, label):
        # For each negative, loss = max(0, margin - (positive_score - negative_score))
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        batch_size = scores.size(0)
        losses = []
        for b in range(batch_size):
            pos_score = scores[b, label[b]]
            for i in range(scores.size(1)):
                if i != label[b].item():
                    neg_score = scores[b, i]
                    losses.append(F.relu(self.margin - (pos_score - neg_score)))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    @classmethod
    def from_config(cls, config: dict) -> 'ContrastiveLoss':
        return cls(
            loss_type=config.get('loss_type', 'infonce'),
            temperature=config.get('temperature', 0.07),
            margin=config.get('margin', 0.2),
        )
