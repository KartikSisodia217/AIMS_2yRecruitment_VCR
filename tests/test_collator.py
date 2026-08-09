"""Test dataset collator."""
import pytest
import torch
from data.schemas import VCRBatch
from data.collator import VCRCollator

def test_collator_produces_batch(debug_dataset):
    collator = VCRCollator(load_images=False)
    samples = [debug_dataset[i] for i in range(3)]
    batch = collator(samples)
    assert isinstance(batch, VCRBatch)

def test_collator_batch_size(debug_dataset):
    collator = VCRCollator(load_images=False)
    samples = [debug_dataset[i] for i in range(3)]
    batch = collator(samples)
    assert len(batch.sample_ids) == 3

def test_collator_answer_labels_shape(debug_dataset):
    collator = VCRCollator(load_images=False)
    samples = [debug_dataset[i] for i in range(3)]
    batch = collator(samples)
    assert batch.answer_labels.shape == (3,)

def test_collator_rationale_labels_shape(debug_dataset):
    collator = VCRCollator(load_images=False)
    samples = [debug_dataset[i] for i in range(3)]
    batch = collator(samples)
    assert batch.rationale_labels.shape == (3,)

def test_collator_answer_choices_shape(debug_dataset):
    collator = VCRCollator(load_images=False)
    samples = [debug_dataset[i] for i in range(3)]
    batch = collator(samples)
    assert len(batch.answer_choices) == 3
    assert len(batch.answer_choices[0]) == 4

def test_collator_handles_no_labels(debug_dataset):
    import copy
    collator = VCRCollator(load_images=False)
    samples = [copy.deepcopy(debug_dataset[i]) for i in range(2)]
    for s in samples:
        s.answer_label = None
        s.rationale_label = None
    batch = collator(samples)
    assert batch.answer_labels is None
    assert batch.rationale_labels is None
