"""VCR PyTorch Dataset."""

import json
import logging
from typing import Optional, List
import torch
from torch.utils.data import Dataset
from .schemas import VCRSample
from .preprocessing import parse_vcr_sample, validate_vcr_raw

logger = logging.getLogger(__name__)

class VCRDataset(Dataset):
    """PyTorch Dataset for VCR JSONL files."""
    
    def __init__(self, jsonl_path: str, vcr_dir: str, 
                 max_samples: Optional[int] = None,
                 reference_format: str = "person N",
                 validate: bool = True):
        """
        Initialize the VCR Dataset.
        
        Args:
            jsonl_path: Path to the VCR jsonl file.
            vcr_dir: Base directory containing 'vcr1images'.
            max_samples: Optional limit for debugging.
            reference_format: Format string for reference resolution.
            validate: Whether to run validation on each loaded sample.
        """
        self.samples: List[VCRSample] = []
        self.invalid_count = 0
        
        logger.info(f"Loading VCR data from {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw = json.loads(line)
                
                if validate:
                    errors = validate_vcr_raw(raw)
                    if errors:
                        logger.warning(f"Skipping invalid raw sample {raw.get('annot_id')}: {errors}")
                        self.invalid_count += 1
                        continue
                        
                sample = parse_vcr_sample(raw, vcr_dir, reference_format)
                
                if validate and not sample.is_valid:
                    logger.warning(f"Skipping parsed sample {sample.sample_id} due to validation errors: {sample.validate()}")
                    self.invalid_count += 1
                    continue
                    
                self.samples.append(sample)
                
                if max_samples and len(self.samples) >= max_samples:
                    break
                    
        logger.info(f"Loaded {len(self.samples)} valid samples. Skipped {self.invalid_count} invalid samples.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VCRSample:
        return self.samples[idx]
    
    def get_statistics(self) -> dict:
        """Return dataset stats: total samples, valid samples, label distributions, etc."""
        ans_labels = [s.answer_label for s in self.samples if s.answer_label is not None]
        rat_labels = [s.rationale_label for s in self.samples if s.rationale_label is not None]
        
        return {
            "total_samples": len(self.samples),
            "invalid_skipped": self.invalid_count,
            "has_answers": len(ans_labels),
            "has_rationales": len(rat_labels),
            "answer_distribution": {k: ans_labels.count(k) for k in set(ans_labels)} if ans_labels else {},
            "rationale_distribution": {k: rat_labels.count(k) for k in set(rat_labels)} if rat_labels else {}
        }
    
    def get_sample_by_id(self, sample_id: str) -> Optional[VCRSample]:
        """Find a sample by its string ID."""
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        return None
