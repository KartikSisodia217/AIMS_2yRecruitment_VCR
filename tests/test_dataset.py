"""Test dataset implementation."""
import pytest

def test_debug_dataset_creates_samples(debug_dataset):
    assert len(debug_dataset) == 5

def test_debug_dataset_samples_valid(debug_dataset):
    for i in range(len(debug_dataset)):
        sample = debug_dataset[i]
        assert sample.is_valid

def test_debug_dataset_answer_choices_count(debug_dataset):
    for i in range(len(debug_dataset)):
        sample = debug_dataset[i]
        assert len(sample.answer_choices) == 4

def test_debug_dataset_rationale_choices_count(debug_dataset):
    for i in range(len(debug_dataset)):
        sample = debug_dataset[i]
        assert len(sample.rationale_choices) == 4

def test_debug_dataset_labels_range(debug_dataset):
    for i in range(len(debug_dataset)):
        sample = debug_dataset[i]
        assert 0 <= sample.answer_label <= 3
        assert 0 <= sample.rationale_label <= 3

def test_debug_dataset_unique_sample_id(debug_dataset):
    ids = [sample.sample_id for sample in debug_dataset]
    assert len(set(ids)) == len(ids)

def test_debug_dataset_indexing(debug_dataset):
    sample = debug_dataset[0]
    assert sample is not None

def test_debug_dataset_len(debug_dataset):
    assert len(debug_dataset) > 0

import json
from src.dataset import VCRDataset

def test_vcr_dataset_custom_image_dir(tmp_path):
    data_dir = tmp_path / "data"
    annots_dir = data_dir / "vcr1annots"
    annots_dir.mkdir(parents=True)
    
    jsonl_path = annots_dir / "train.jsonl"
    fake_record = {
        "img_fn": "movie1/scene1.jpg",
        "metadata_fn": "movie1/scene1.json",
        "question": ["What", "is", "this", "?"],
        "answer_choices": [["A", "dog"]],
        "rationale_choices": [["Because", "it", "barks"]]
    }
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(fake_record) + "\n")
        
    custom_image_dir = tmp_path / "custom_images"
    custom_image_dir.mkdir(parents=True)
    
    dataset = VCRDataset(split='train', data_dir=str(data_dir), image_dir=str(custom_image_dir))
    
    assert str(dataset.image_dir) == str(custom_image_dir)
    
    sample = dataset[0]
    expected_img_path = custom_image_dir / "movie1" / "scene1.jpg"
    
    # On Windows, path separators might mismatch if we compare string representation directly,
    # so we should compare using pathlib.Path or just checking if custom_image_dir string is in the path
    import pathlib
    assert pathlib.Path(sample["image_path"]) == expected_img_path
