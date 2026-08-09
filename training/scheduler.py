"""Scheduler building utility."""

import torch

def build_scheduler(optimizer, config: dict, num_training_steps: int):
    """Build LR scheduler."""
    warmup_ratio = config.get('warmup_ratio', 0.1)
    warmup_steps = int(num_training_steps * warmup_ratio)
    
    # Linear warmup + linear decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                   float(max(1, num_training_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
