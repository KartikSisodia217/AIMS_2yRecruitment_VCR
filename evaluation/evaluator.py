"""VCR Evaluator implementation."""

import os
from typing import Optional
from .metrics import compute_all_metrics

class VCREvaluator:
    """Evaluates a CACRSPModel on VCR data."""
    
    def __init__(self, model, config: dict = None):
        self.model = model
        self.config = config or {}
    
    def evaluate(self, dataset, use_gt_answer_for_rationale: bool = False,
                 max_samples: Optional[int] = None) -> dict:
        """Run evaluation on dataset.
        
        Returns dict with all metrics.
        """
        # Collect predictions
        answer_preds = []
        answer_labels = []
        rationale_preds = []
        rationale_labels = []
        rationale_preds_gt = []  # with GT answer
        
        samples = dataset
        if max_samples:
            samples = [dataset[i] for i in range(min(max_samples, len(dataset)))]
        
        for sample in samples:
            if not sample.has_labels:
                continue
            
            # Load image
            image = self._load_image(sample.image_path)
            
            # Full pipeline prediction (predicted answer → predicted rationale)
            result = self.model.predict(
                image, sample.question, sample.answer_choices,
                sample.rationale_choices
            )
            
            answer_preds.append(result['predicted_answer'])
            answer_labels.append(sample.answer_label)
            rationale_preds.append(result['predicted_rationale'])
            rationale_labels.append(sample.rationale_label)
            
            # Also evaluate with GT answer for QA→R
            if use_gt_answer_for_rationale:
                gt_result = self.model.predict(
                    image, sample.question, sample.answer_choices,
                    sample.rationale_choices,
                    use_gt_answer=True, gt_answer_idx=sample.answer_label
                )
                rationale_preds_gt.append(gt_result['predicted_rationale'])
        
        return compute_all_metrics(
            answer_preds, answer_labels,
            rationale_preds, rationale_labels,
            rationale_preds_gt if rationale_preds_gt else None,
            rationale_labels if rationale_preds_gt else None,
        )
    
    def _load_image(self, path):
        """Load image, returning None if path doesn't exist."""
        if os.path.exists(path):
            from PIL import Image
            return Image.open(path).convert('RGB')
        return None  # Mock VLM handles None gracefully
