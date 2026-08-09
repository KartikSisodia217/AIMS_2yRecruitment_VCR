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
