"""Training loop implementation."""

import os
from typing import Optional
from PIL import Image

class Trainer:
    """Training loop for CACR-SP VCR models."""
    
    def __init__(self, model, train_dataset, val_dataset, loss_fn, config, evaluator=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn
        self.config = config
        self.evaluator = evaluator
        
        # Avoid circular import by using built-in methods if optimizer isn't passed
        from .optimizer import build_optimizer
        self.optimizer = build_optimizer(model, config.get('training', {}))
        
        # Basic setup
        self.current_step = 0
        self.current_epoch = 0
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    
    def _load_image(self, path):
        if os.path.exists(path):
            return Image.open(path).convert('RGB')
        return None
        
    def train_one_step(self, sample) -> dict:
        """Execute one training step on a single sample. Returns loss dict."""
        self.model.train()
        
        if not sample.has_labels:
            return {}
            
        image = self._load_image(sample.image_path)
        
        # Forward pass
        # This is a simplified forward pass; in reality, we need batched processing.
        # But for CPU processing or simple loop, we do it per sample.
        result = self.model.forward(
            image=image,
            question=sample.question,
            answer_choices=sample.answer_choices,
            rationale_choices=sample.rationale_choices,
        )
        
        import torch
        answer_label = torch.tensor(sample.answer_label).to(result['answer_scores'].device)
        rationale_label = torch.tensor(sample.rationale_label).to(result['rationale_scores'].device)
        
        loss_dict = self.loss_fn(
            rationale_scores=result['rationale_scores'],
            rationale_label=rationale_label,
            blind_scores=result.get('blind_rationale_scores'),
            normal_scores=result.get('rationale_scores'),
            answer_scores=result['answer_scores'],
            answer_label=answer_label
        )
        
        loss = loss_dict['total'] / self.gradient_accumulation_steps
        loss.backward()
        
        self.current_step += 1
        
        if self.current_step % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return {k: v.item() for k, v in loss_dict.items() if isinstance(v, torch.Tensor)}
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch. Returns averaged metrics."""
        self.current_epoch = epoch
        metrics_sum = {}
        count = 0
        
        for sample in self.train_dataset:
            metrics = self.train_one_step(sample)
            if not metrics:
                continue
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            count += 1
            
        if count == 0:
            return {}
        return {k: v / count for k, v in metrics_sum.items()}
    
    def train(self, num_epochs: int = 1) -> dict:
        """Full training loop. Returns final metrics."""
        final_metrics = {}
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch} metrics: {epoch_metrics}")
            
            if self.evaluator and self.val_dataset:
                val_metrics = self.validate()
                print(f"Epoch {epoch} validation: {val_metrics}")
                final_metrics = val_metrics
        return final_metrics
    
    def validate(self) -> dict:
        """Run validation. Returns metrics."""
        if not self.evaluator or not self.val_dataset:
            return {}
        
        self.model.eval()
        return self.evaluator.evaluate(self.val_dataset)
