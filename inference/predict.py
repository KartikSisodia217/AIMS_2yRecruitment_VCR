"""Prediction functions for VCR models."""

import os
from typing import List, Any

def predict_sample(model, sample: Any, load_image: bool = True) -> dict:
    """Run inference on a single VCR sample.
    
    Args:
        model: VCR Model instance.
        sample: VCRSample object.
        load_image: Whether to load image from disk.
        
    Returns:
        dict with: predicted_answer, predicted_rationale, answer_scores,
        rationale_scores, is_answer_correct, is_rationale_correct, is_joint_correct
    """
    image = None
    if load_image and os.path.exists(sample.image_path):
        from PIL import Image
        image = Image.open(sample.image_path).convert('RGB')
    
    result = model.predict(image, sample.question, sample.answer_choices, sample.rationale_choices)
    
    # Add correctness info if labels available
    if hasattr(sample, 'has_labels') and sample.has_labels:
        result['is_answer_correct'] = result['predicted_answer'] == sample.answer_label
        result['is_rationale_correct'] = result['predicted_rationale'] == sample.rationale_label
        result['is_joint_correct'] = result['is_answer_correct'] and result['is_rationale_correct']
    
    return result


def predict_batch(model, samples: List[Any], load_images: bool = True) -> List[dict]:
    """Run inference on multiple samples."""
    return [predict_sample(model, s, load_images) for s in samples]
