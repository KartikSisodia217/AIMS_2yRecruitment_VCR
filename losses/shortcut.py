"""Shortcut penalty for visual-agnostic reasoning reduction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ShortcutPenalty(nn.Module):
    """Visual-Agnostic Shortcut Penalty (VASP) — PLACEHOLDER.
    
    Penalizes the model when it can solve the task without visual information.
    
    Conceptual goal:
        normal_performance → high (good, model uses vision)
        blind_performance → low (good, model can't cheat)
    
    If blind_performance is high, the model may be exploiting text shortcuts.
    
    IMPORTANT: This is a baseline/placeholder formulation.
    The exact mathematical formulation is an OPEN RESEARCH QUESTION.
    Do not treat this implementation as the final research formulation.
    
    Current placeholder:
        L_SP = max(0, blind_confidence - margin)
    
    where blind_confidence is the softmax probability assigned to the
    correct answer by the blind (no-image) branch.
    
    Alternatives to investigate:
    - KL divergence between normal and blind distributions
    - Difference in ranking positions
    - Mutual information estimates
    - Gradient-based penalties
    
    # RESEARCH_DECISION: exact SP formulation
    # RESEARCH_DECISION: should SP operate on logits, probs, similarities, or ranks
    # RESEARCH_DECISION: should SP apply to answer stage, rationale stage, or both
    """
    
    def __init__(self, lambda_sp: float = 0.1, margin: float = 0.25,
                 apply_to: str = 'rationale',
                 formulation: str = 'confidence_penalty'):
        super().__init__()
        self.lambda_sp = lambda_sp
        self.margin = margin
        self.apply_to = apply_to  # 'answer', 'rationale', 'both'
        self.formulation = formulation  # 'confidence_penalty', 'kl_divergence'
    
    def forward(self, blind_scores: torch.Tensor, 
                label: torch.Tensor,
                normal_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute shortcut penalty.
        
        Args:
            blind_scores: Shape [batch_size, num_candidates] or [num_candidates]
                Scores from the blind (no-image) branch.
            label: Shape [batch_size] or scalar. Correct label index.
            normal_scores: Optional. Shape matching blind_scores.
                Scores from the normal (with-image) branch.
                Required for some formulations (e.g., KL divergence).
        
        Returns:
            Scalar penalty tensor (to be multiplied by lambda_sp externally or internally)
        """
        if self.formulation == 'confidence_penalty':
            return self._confidence_penalty(blind_scores, label)
        elif self.formulation == 'kl_divergence':
            return self._kl_divergence(blind_scores, normal_scores)
        else:
            raise ValueError(f"Unknown formulation: {self.formulation}")
    
    def _confidence_penalty(self, blind_scores, label):
        """Penalize when blind branch is confident about correct answer."""
        if blind_scores.dim() == 1:
            blind_scores = blind_scores.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        blind_probs = F.softmax(blind_scores, dim=-1)
        # Get probability of correct label under blind branch
        blind_confidence = blind_probs.gather(1, label.unsqueeze(1)).squeeze(1)
        # Penalize if confidence exceeds margin (i.e., model can solve without vision)
        penalty = F.relu(blind_confidence - self.margin)
        return penalty.mean()
    
    def _kl_divergence(self, blind_scores, normal_scores):
        """KL(blind || uniform) — penalize blind branch for being non-uniform."""
        if blind_scores is None:
            return torch.tensor(0.0)
        if blind_scores.dim() == 1:
            blind_scores = blind_scores.unsqueeze(0)
        
        blind_log_probs = F.log_softmax(blind_scores, dim=-1)
        num_candidates = blind_scores.size(-1)
        uniform = torch.ones_like(blind_scores) / num_candidates
        # KL(blind || uniform) — penalize when blind is far from uniform
        kl = F.kl_div(uniform.log(), blind_log_probs.exp(), reduction='batchmean')
        return kl
    
    @classmethod
    def from_config(cls, config: dict) -> 'ShortcutPenalty':
        return cls(
            lambda_sp=config.get('lambda_sp', 0.1),
            margin=config.get('margin', 0.25),
            apply_to=config.get('apply_to', 'rationale'),
            formulation=config.get('formulation', 'confidence_penalty'),
        )
