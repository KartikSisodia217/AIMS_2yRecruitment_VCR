"""Checkpointing utility."""

import os
import glob
import torch
from typing import Optional, List

class CheckpointManager:
    """Manages saving and loading of training checkpoints."""
    
    def __init__(self, output_dir: str, experiment_name: str):
        self.checkpoint_dir = os.path.join(output_dir, experiment_name, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save(self, model, optimizer, scheduler, epoch, step, metrics, config):
        """Save checkpoint."""
        path = os.path.join(self.checkpoint_dir, f'checkpoint_step{step}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'config': config,
        }, path)
        return path
    
    def load(self, path, model, optimizer=None, scheduler=None):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    
    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        # Sort by step number
        def extract_step(path):
            filename = os.path.basename(path)
            # e.g., 'checkpoint_step100.pt'
            step_str = filename.replace('checkpoint_step', '').replace('.pt', '')
            try:
                return int(step_str)
            except ValueError:
                return -1
        checkpoints.sort(key=extract_step)
        return checkpoints[-1]
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_step*.pt')
        return glob.glob(pattern)
