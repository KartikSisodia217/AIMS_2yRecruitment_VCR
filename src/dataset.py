import json
import re
import os
from pathlib import Path
from torch.utils.data import Dataset

def parse_tokenized_text(tokenized_text):
    """
    Parses a tokenized text array into a readable string.
    e.g., ["Does", [2], "feel", "comfortable", "?"] -> "Does [2] feel comfortable?"
    """
    words = []
    for item in tokenized_text:
        if isinstance(item, list):
            words.append(str(item))
        else:
            words.append(str(item))
            
    text = " ".join(words)
    # Basic cleanup for punctuation spacing
    text = re.sub(r' \?', '?', text)
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' ,', ',', text)
    return text

def clean_path(path):
    if not isinstance(path, str):
        return path
    # Removes markdown/mailto artifacts like [filename.jpg](mailto:filename.jpg)
    # while preserving the rest of the directory path.
    return re.sub(r'\[(.*?)\]\(mailto:.*?\)', r'\1', path)

class VCRDataset(Dataset):
    def __init__(self, split='train', data_dir='data/vcr'):
        """
        Initializes the VCRDataset.
        :param split: 'train', 'val', or 'test'
        :param data_dir: Root directory of the VCR data
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.jsonl_path = self.data_dir / 'vcr1annots' / f'{split}.jsonl'
        self.image_dir = self.data_dir / 'vcr1images'
        
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.jsonl_path}")
            
        # Avoid loading the entire JSONL into memory by storing line byte offsets.
        self.line_offsets = []
        with open(self.jsonl_path, 'rb') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)
                
    def __len__(self):
        return len(self.line_offsets)
        
    def __getitem__(self, idx):
        # Open file dynamically so it's safe for multi-processing (num_workers > 0)
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            
        record = json.loads(line)
        
        # Clean paths to fix markdown corruption
        img_fn = clean_path(record.get('img_fn', ''))
        metadata_fn = clean_path(record.get('metadata_fn', ''))
        
        # Resolve absolute paths
        image_path = self.image_dir / img_fn
        metadata_path = self.image_dir / metadata_fn
        
        # Format the question, answer choices, and rationale choices
        question = parse_tokenized_text(record.get('question', []))
        
        answer_choices = []
        for choice in record.get('answer_choices', []):
            answer_choices.append(parse_tokenized_text(choice))
            
        rationale_choices = []
        for choice in record.get('rationale_choices', []):
            rationale_choices.append(parse_tokenized_text(choice))
            
        sample = {
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "objects": record.get("objects", []),
            "question": question,
            "answer_choices": answer_choices,
            "answer_label": record.get("answer_label", -1),
            "rationale_choices": rationale_choices,
            "rationale_label": record.get("rationale_label", -1)
        }
        
        return sample
        
    def load_metadata(self, metadata_path):
        """
        Loads the metadata JSON when requested.
        Returns a dict exposing: boxes, segms, width, height.
        """
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        return {
            "boxes": meta.get("boxes", []),
            "segms": meta.get("segms", []),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0)
        }
