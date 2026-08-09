"""Data preprocessing for VCR."""

import os
import json
from typing import List, Dict, Optional, Any
from .schemas import VCRSample

def build_object_name_map(objects: List[str]) -> Dict[int, str]:
    """
    Build a mapping from object index to a formatted name.
    
    Args:
        objects: List of object categories, e.g. ["person", "person", "car"]
        
    Returns:
        Dict mapping index to name, e.g. {0: "person 1", 1: "person 2", 2: "car 1"}
    """
    counts = {}
    obj_map = {}
    for i, obj in enumerate(objects):
        counts[obj] = counts.get(obj, 0) + 1
        obj_map[i] = f"{obj} {counts[obj]}"
    return obj_map

def format_reference(names: List[str], format_type: str = "person N") -> str:
    """Format a list of resolved object names according to the requested format."""
    if not names:
        return ""
        
    if format_type == "tagged":
        formatted = [f"<{name.replace(' ', '_')}>" for name in names]
    elif format_type == "bracketed":
        formatted = [f"[{name}]" for name in names]
    else:  # default: person N
        formatted = names
        
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"

def resolve_references(tokens: List[Any], objects: List[str], format: str = "person N") -> str:
    """
    Resolve references in a VCR token list.
    
    Args:
        tokens: Token list, e.g. ["Why", "is", [0], "looking", "at", [1], "?"]
        objects: Object categories
        format: Format string
        
    Returns:
        Resolved text.
    """
    obj_map = build_object_name_map(objects)
    resolved_parts = []
    
    for token in tokens:
        if isinstance(token, list):
            # It's a reference list like [0] or [0, 1]
            names = [obj_map.get(idx, f"object {idx}") for idx in token]
            resolved_parts.append(format_reference(names, format))
        else:
            resolved_parts.append(str(token))
            
    # Basic space joining (can be improved with better detokenization if needed)
    text = " ".join(resolved_parts)
    # Fix punctuation spacing
    for punc in [".", ",", "?", "!"]:
        text = text.replace(f" {punc}", punc)
    return text

def validate_vcr_raw(raw: Dict[str, Any]) -> List[str]:
    """Validate a raw VCR JSON dictionary."""
    errors = []
    required = ["question", "answer_choices", "rationale_choices", "img_fn", "objects"]
    for req in required:
        if req not in raw:
            errors.append(f"Missing required field: {req}")
            
    if "answer_choices" in raw and len(raw["answer_choices"]) != 4:
        errors.append("answer_choices must have exactly 4 items")
    if "rationale_choices" in raw and len(raw["rationale_choices"]) != 4:
        errors.append("rationale_choices must have exactly 4 items")
        
    return errors

def parse_vcr_sample(raw: Dict[str, Any], vcr_dir: str, reference_format: str = "person N") -> VCRSample:
    """
    Convert a raw VCR dictionary into a VCRSample object.
    
    Args:
        raw: Raw JSON dictionary
        vcr_dir: Base directory of VCR data (should contain vcr1images/)
        reference_format: Formatting style for object references
        
    Returns:
        VCRSample object
    """
    objects = raw.get("objects", [])
    
    question = resolve_references(raw["question"], objects, reference_format)
    answers = [resolve_references(ans, objects, reference_format) for ans in raw["answer_choices"]]
    rationales = [resolve_references(rat, objects, reference_format) for rat in raw["rationale_choices"]]
    
    # Path resolution
    img_path = ""
    if "img_fn" in raw and vcr_dir:
        img_path = os.path.join(vcr_dir, "vcr1images", raw["img_fn"])
        
    # Optional metadata loading
    bboxes = None
    metadata = raw.copy()
    if "metadata_fn" in raw and vcr_dir:
        meta_path = os.path.join(vcr_dir, "vcr1images", raw["metadata_fn"])
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                    if "boxes" in meta_data:
                        bboxes = meta_data["boxes"]
                    metadata.update(meta_data)
            except Exception as e:
                # Log or handle exception as needed
                pass
                
    return VCRSample(
        sample_id=raw.get("annot_id", "unknown"),
        image_path=img_path,
        question=question,
        answer_choices=answers,
        rationale_choices=rationales,
        answer_label=raw.get("answer_label"),
        rationale_label=raw.get("rationale_label"),
        objects=objects,
        bboxes=bboxes,
        metadata=metadata,
        raw_question=raw["question"],
        raw_answer_choices=raw["answer_choices"],
        raw_rationale_choices=raw["rationale_choices"]
    )
