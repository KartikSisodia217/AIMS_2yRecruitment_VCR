import os
import zipfile
import io
import random
import numpy as np
import torch
from PIL import Image

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_image(image_path, zip_path="data/vcr/vcr1images.zip"):
    """
    Loads an image either from the extracted directory or from the zip archive.
    """
    if os.path.exists(image_path):
        return Image.open(image_path).convert("RGB")
        
    # Read from zip if not extracted
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Image not found at {image_path} and zip archive {zip_path} is missing.")
        
    img_path_normalized = image_path.replace("\\", "/")
    if "vcr1images/" in img_path_normalized:
        rel_path = "vcr1images/" + img_path_normalized.split("vcr1images/")[-1]
    else:
        raise ValueError(f"Cannot parse relative path from {image_path}")
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(rel_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

def vcr_collate_fn(batch):
    """
    Custom collate function to prevent default_collate from mangling lists of strings.
    Returns a dictionary of lists.
    """
    return {
        "image_path": [b["image_path"] for b in batch],
        "metadata_path": [b["metadata_path"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "question": [b["question"] for b in batch],
        "answer_choices": [b["answer_choices"] for b in batch],
        "answer_label": torch.tensor([b["answer_label"] for b in batch], dtype=torch.long),
        "rationale_choices": [b["rationale_choices"] for b in batch],
        "rationale_label": torch.tensor([b["rationale_label"] for b in batch], dtype=torch.long),
    }
