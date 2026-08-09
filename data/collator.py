"""Collator for VCR DataLoader."""

from typing import List, Any
import torch
from PIL import Image
from .schemas import VCRSample, VCRBatch

class VCRCollator:
    """Collates VCRSample objects into VCRBatch."""
    
    def __init__(self, load_images: bool = True):
        self.load_images = load_images
    
    def __call__(self, samples: List[VCRSample]) -> VCRBatch:
        sample_ids = []
        images = []
        questions = []
        answer_choices = []
        rationale_choices = []
        answer_labels = []
        rationale_labels = []
        objects = []
        bboxes = []
        metadata = []
        
        for sample in samples:
            sample_ids.append(sample.sample_id)
            
            if self.load_images:
                try:
                    img = Image.open(sample.image_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    # Fallback to empty image or raise error depending on needs
                    # For now, append None (downstream must handle)
                    images.append(None)
            else:
                images.append(sample.image_path)
                
            questions.append(sample.question)
            answer_choices.append(sample.answer_choices)
            rationale_choices.append(sample.rationale_choices)
            
            answer_labels.append(sample.answer_label)
            rationale_labels.append(sample.rationale_label)
            
            objects.append(sample.objects)
            bboxes.append(sample.bboxes)
            metadata.append(sample.metadata)
            
        # Convert labels to tensors if they exist
        if all(l is not None for l in answer_labels):
            ans_tensor = torch.tensor(answer_labels, dtype=torch.long)
        else:
            ans_tensor = None
            
        if all(l is not None for l in rationale_labels):
            rat_tensor = torch.tensor(rationale_labels, dtype=torch.long)
        else:
            rat_tensor = None
            
        return VCRBatch(
            sample_ids=sample_ids,
            images=images,
            questions=questions,
            answer_choices=answer_choices,
            rationale_choices=rationale_choices,
            answer_labels=ans_tensor,
            rationale_labels=rat_tensor,
            objects=objects,
            bboxes=bboxes,
            metadata=metadata
        )
