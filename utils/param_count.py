"""Parameter counting utilities for PyTorch models."""

import torch.nn as nn
from typing import Dict, Any

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count the number of trainable and non-trainable parameters in a model.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        Dict[str, Any]: Dictionary containing total, trainable, frozen counts and percentage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    trainable_pct = (trainable_params / total_params * 100) if total_params > 0 else 0.0
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_pct': trainable_pct
    }

def print_parameter_summary(model: nn.Module, name: str = "Model") -> None:
    """
    Pretty-print the parameter counts of a model.
    
    Args:
        model (nn.Module): The PyTorch model.
        name (str): The name to display in the summary.
    """
    counts = count_parameters(model)
    print(f"--- {name} Parameter Summary ---")
    print(f"Total parameters:     {counts['total']:,}")
    print(f"Trainable parameters: {counts['trainable']:,}")
    print(f"Frozen parameters:    {counts['frozen']:,}")
    print(f"% Trainable:          {counts['trainable_pct']:.2f}%")
    print("-" * (len(name) + 26))

def get_parameter_groups(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """
    Get parameter counts for each named child module.
    
    Args:
        model (nn.Module): The PyTorch model.
        
    Returns:
        Dict[str, Dict[str, int]]: Dictionary mapping child names to their parameter counts.
    """
    groups = {}
    for name, child in model.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        groups[name] = {
            'total': total,
            'trainable': trainable
        }
    return groups
