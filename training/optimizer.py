"""Optimizer building utility."""

import torch
import torch.nn as nn

def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer with proper parameter groups.
    
    Separates:
    - VLM parameters (if trainable, lower LR)
    - Projection head parameters
    - Other task-specific parameters
    """
    lr = config.get('learning_rate', 2e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    # For now, simple AdamW on all trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Return a dummy optimizer if nothing is trainable
        # (useful for evaluation-only mode)
        return torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=lr)
    
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
