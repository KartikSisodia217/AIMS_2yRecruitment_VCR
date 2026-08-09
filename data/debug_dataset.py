"""Synthetic debug dataset for testing the pipeline without real VCR data."""

import os
import json
import tempfile
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from .schemas import VCRSample

def create_debug_vcr_sample(idx: int = 0, image_path: str = "") -> VCRSample:
    """Create a single synthetic VCR sample."""
    return VCRSample(
        sample_id=f"debug-{idx}",
        image_path=image_path,
        question=f"What is person 1 doing to person 2 in {idx}?",
        answer_choices=[
            "Smiling.",
            "Talking.",
            "Walking away.",
            "Ignoring them."
        ],
        rationale_choices=[
            "Because they are happy.",
            "Because they are engaged.",
            "Because they are leaving.",
            "Because they are rude."
        ],
        answer_label=0,
        rationale_label=0,
        objects=["person", "person", "table"],
        bboxes=[
            [0.0, 0.0, 10.0, 10.0, 1.0],
            [10.0, 10.0, 20.0, 20.0, 0.9],
            [5.0, 5.0, 15.0, 15.0, 0.8]
        ],
        metadata={"debug": True}
    )

def create_debug_jsonl(output_dir: str, num_samples: int = 5) -> str:
    """Creates a JSONL file matching VCR format with synthetic data."""
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "debug_vcr.jsonl")
    
    img_dir = os.path.join(output_dir, "vcr1images", "debug")
    os.makedirs(img_dir, exist_ok=True)
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            img_fn = f"debug/img_{i}.png"
            img_path = os.path.join(output_dir, "vcr1images", img_fn)
            
            # Create a small solid color PNG
            img = Image.new("RGB", (64, 64), color=(i * 10 % 255, 100, 150))
            img.save(img_path)
            
            # Create raw JSON representation
            raw = {
                "annot_id": f"debug-{i}",
                "img_fn": img_fn,
                "objects": ["person", "person", "table"],
                "question": ["What", "is", [0], "doing", "to", [1], "?"],
                "answer_choices": [
                    ["Smiling", "."],
                    ["Talking", "."],
                    ["Walking", "away", "."],
                    ["Ignoring", "them", "."]
                ],
                "rationale_choices": [
                    ["Because", "they", "are", "happy", "."],
                    ["Because", "they", "are", "engaged", "."],
                    ["Because", "they", "are", "leaving", "."],
                    ["Because", "they", "are", "rude", "."]
                ],
                "answer_label": 0,
                "rationale_label": 0
            }
            f.write(json.dumps(raw) + "\n")
            
    return jsonl_path

class DebugVCRDataset(Dataset):
    """In-memory dataset of synthetic VCR samples for testing."""
    
    def __init__(self, num_samples: int = 5, create_images: bool = True, image_dir: Optional[str] = None):
        self.samples = []
        
        if create_images and not image_dir:
            # Create a persistent temp dir for this dataset instance
            self.temp_dir = tempfile.mkdtemp()
            image_dir = self.temp_dir
        
        for i in range(num_samples):
            img_path = ""
            if create_images and image_dir:
                img_path = os.path.join(image_dir, f"debug_img_{i}.png")
                img = Image.new("RGB", (64, 64), color=(i * 20 % 255, 120, 200))
                img.save(img_path)
                
            sample = create_debug_vcr_sample(i, img_path)
            self.samples.append(sample)
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> VCRSample:
        return self.samples[idx]
