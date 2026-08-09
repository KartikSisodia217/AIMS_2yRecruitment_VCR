"""Test evaluation metrics."""
import pytest
from evaluation.metrics import compute_q_to_a, compute_qa_to_r, compute_q_to_ar

def test_compute_q_to_a_all_correct():
    preds = [0, 1, 2, 3]
    labels = [0, 1, 2, 3]
    assert compute_q_to_a(preds, labels) == 1.0

def test_compute_q_to_a_all_wrong():
    preds = [1, 2, 3, 0]
    labels = [0, 1, 2, 3]
    assert compute_q_to_a(preds, labels) == 0.0

def test_compute_q_to_a_half_correct():
    preds = [0, 1, 3, 0]
    labels = [0, 1, 2, 3]
    assert compute_q_to_a(preds, labels) == 0.5

def test_compute_qa_to_r_all_correct():
    preds = [0, 1, 2, 3]
    labels = [0, 1, 2, 3]
    assert compute_qa_to_r(preds, labels) == 1.0

def test_compute_qa_to_r_all_wrong():
    preds = [1, 2, 3, 0]
    labels = [0, 1, 2, 3]
    assert compute_qa_to_r(preds, labels) == 0.0

def test_compute_q_to_ar_both_correct():
    a_preds = [0]
    a_labels = [0]
    r_preds = [0]
    r_labels = [0]
    assert compute_q_to_ar(a_preds, a_labels, r_preds, r_labels) == 1.0

def test_compute_q_to_ar_answer_correct_rationale_wrong():
    a_preds = [0]
    a_labels = [0]
    r_preds = [1]
    r_labels = [0]
    assert compute_q_to_ar(a_preds, a_labels, r_preds, r_labels) == 0.0

def test_compute_q_to_ar_answer_wrong_rationale_correct():
    a_preds = [1]
    a_labels = [0]
    r_preds = [0]
    r_labels = [0]
    assert compute_q_to_ar(a_preds, a_labels, r_preds, r_labels) == 0.0

def test_compute_q_to_ar_both_wrong():
    a_preds = [1]
    a_labels = [0]
    r_preds = [1]
    r_labels = [0]
    assert compute_q_to_ar(a_preds, a_labels, r_preds, r_labels) == 0.0

def test_compute_q_to_ar_mixed_batch():
    # GT: A2,R3 | A1,R0 | A3,R2 | A0,R1
    # Pred: A2,R3 | A1,R0 | A3,R1 | A0,R0
    a_preds = [2, 1, 3, 0]
    a_labels = [2, 1, 3, 0]
    
    r_preds = [3, 0, 1, 0]
    r_labels = [3, 0, 2, 1]
    
    assert compute_q_to_a(a_preds, a_labels) == 1.0
    assert compute_q_to_ar(a_preds, a_labels, r_preds, r_labels) == 0.5
