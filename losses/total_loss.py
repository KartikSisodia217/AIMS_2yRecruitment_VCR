"""Combined loss module."""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .contrastive import ContrastiveLoss
from .shortcut import ShortcutPenalty

class TotalLoss(nn.Module):
    """Combined loss: Contrastive + lambda * Shortcut Penalty.
    
    Returns a dict with all individual losses for logging.
    """
    
    def __init__(self, contrastive_loss: ContrastiveLoss,
                 shortcut_penalty: Optional[ShortcutPenalty] = None,
                 lambda_sp: float = 0.1,
                 answer_loss: Optional[nn.Module] = None,
                 lambda_answer: float = 1.0):
        super().__init__()
        self.contrastive_loss = contrastive_loss
        self.shortcut_penalty = shortcut_penalty
        self.lambda_sp = lambda_sp
        self.answer_loss = answer_loss
        self.lambda_answer = lambda_answer
    
    def forward(self, rationale_scores: torch.Tensor,
                rationale_label: torch.Tensor,
                blind_scores: Optional[torch.Tensor] = None,
                normal_scores: Optional[torch.Tensor] = None,
                answer_scores: Optional[torch.Tensor] = None,
                answer_label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute total loss.
        
        Returns:
            Dict with keys: 'total', 'contrastive', 'shortcut', 'answer'
            Each value is a scalar tensor.
        """
        result = {}
        
        # Contrastive loss on rationale scores
        contrastive = self.contrastive_loss(rationale_scores, rationale_label)
        result['contrastive'] = contrastive
        total = contrastive
        
        # Shortcut penalty (optional)
        if self.shortcut_penalty is not None and blind_scores is not None:
            sp = self.shortcut_penalty(blind_scores, rationale_label, normal_scores)
            result['shortcut'] = sp
            total = total + self.lambda_sp * sp
        else:
            result['shortcut'] = torch.tensor(0.0)
        
        # Answer loss (optional)
        if self.answer_loss is not None and answer_scores is not None and answer_label is not None:
            ans_loss = self.answer_loss(answer_scores, answer_label)
            result['answer'] = ans_loss
            total = total + self.lambda_answer * ans_loss
        else:
            result['answer'] = torch.tensor(0.0)
        
        result['total'] = total
        return result
    
    @classmethod
    def from_config(cls, config: dict) -> 'TotalLoss':
        contrastive = ContrastiveLoss.from_config(config.get('contrastive', {}))
        
        sp_config = config.get('shortcut', {})
        shortcut = None
        if sp_config.get('enabled', False):
            shortcut = ShortcutPenalty.from_config(sp_config)
        
        return cls(
            contrastive_loss=contrastive,
            shortcut_penalty=shortcut,
            lambda_sp=sp_config.get('lambda_sp', 0.1),
        )
