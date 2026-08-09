"""Data schemas for VCR."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

try:
    import torch
    TensorType = torch.Tensor
except ImportError:
    TensorType = Any

@dataclass
class VCRSample:
    """Normalized VCR sample. The rest of the codebase uses this, never raw JSON."""
    sample_id: str
    image_path: str  # absolute path to image file
    question: str  # resolved human-readable question text
    answer_choices: List[str]  # exactly 4 resolved answer texts
    rationale_choices: List[str]  # exactly 4 resolved rationale texts
    answer_label: Optional[int] = None  # 0-3, None for test split
    rationale_label: Optional[int] = None  # 0-3, None for test split
    objects: List[str] = field(default_factory=list)  # e.g. ["person", "person", "car"]
    bboxes: Optional[List[List[float]]] = None  # [x1, y1, x2, y2, score] per object
    metadata: Dict[str, Any] = field(default_factory=dict)  # any extra info
    
    # Raw VCR format (preserved for debugging/alternative processing)
    raw_question: Optional[List] = None
    raw_answer_choices: Optional[List[List]] = None
    raw_rationale_choices: Optional[List[List]] = None
    
    def validate(self) -> List[str]:
        """Validate sample integrity. Returns list of error messages (empty if valid)."""
        errors = []
        if not self.sample_id:
            errors.append("Missing sample_id")
        if not self.image_path:
            errors.append("Missing image_path")
        if not self.question:
            errors.append("Empty question")
        if len(self.answer_choices) != 4:
            errors.append(f"Expected 4 answer choices, got {len(self.answer_choices)}")
        if len(self.rationale_choices) != 4:
            errors.append(f"Expected 4 rationale choices, got {len(self.rationale_choices)}")
        if self.answer_label is not None and not (0 <= self.answer_label <= 3):
            errors.append(f"Invalid answer_label: {self.answer_label}")
        if self.rationale_label is not None and not (0 <= self.rationale_label <= 3):
            errors.append(f"Invalid rationale_label: {self.rationale_label}")
        return errors
    
    @property
    def is_valid(self) -> bool:
        return len(self.validate()) == 0
    
    @property
    def has_labels(self) -> bool:
        return self.answer_label is not None and self.rationale_label is not None


@dataclass
class VCRBatch:
    """Batched VCR samples ready for model consumption."""
    sample_ids: List[str]
    images: Any  # PIL Images or tensors
    questions: List[str]
    answer_choices: List[List[str]]  # [batch_size, 4]
    rationale_choices: List[List[str]]  # [batch_size, 4]
    answer_labels: Optional[TensorType] = None  # [batch_size]
    rationale_labels: Optional[TensorType] = None  # [batch_size]
    objects: Optional[List[List[str]]] = None
    bboxes: Optional[List[Optional[List[List[float]]]]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
