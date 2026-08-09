"""Loss functions for VCR tasks."""

from .classification import CrossEntropyScorer
from .contrastive import ContrastiveLoss
from .shortcut import ShortcutPenalty
from .total_loss import TotalLoss

__all__ = ['CrossEntropyScorer', 'ContrastiveLoss', 'ShortcutPenalty', 'TotalLoss']
