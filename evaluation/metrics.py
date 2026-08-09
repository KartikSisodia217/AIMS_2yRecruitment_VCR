"""VCR metrics computation."""

from typing import List

def compute_q_to_a(answer_preds: List[int], answer_labels: List[int]) -> float:
    """Compute Q→A accuracy: fraction of correct answer predictions.
    
    Args:
        answer_preds: List of predicted answer indices (0-3)
        answer_labels: List of ground-truth answer indices (0-3)
    
    Returns:
        Accuracy in [0, 1]
    """
    assert len(answer_preds) == len(answer_labels), "Length mismatch"
    if len(answer_preds) == 0:
        return 0.0
    correct = sum(p == l for p, l in zip(answer_preds, answer_labels))
    return correct / len(answer_preds)


def compute_qa_to_r(rationale_preds: List[int], rationale_labels: List[int]) -> float:
    """Compute QA→R accuracy: fraction of correct rationale predictions.
    
    NOTE: This metric assumes the GROUND-TRUTH answer was used for Stage 2.
    It isolates rationale performance from answer performance.
    """
    assert len(rationale_preds) == len(rationale_labels)
    if len(rationale_preds) == 0:
        return 0.0
    correct = sum(p == l for p, l in zip(rationale_preds, rationale_labels))
    return correct / len(rationale_preds)


def compute_q_to_ar(answer_preds: List[int], answer_labels: List[int],
                    rationale_preds: List[int], rationale_labels: List[int]) -> float:
    """Compute Q→AR accuracy: fraction where BOTH answer AND rationale are correct.
    
    A prediction is correct only if:
        predicted_answer == ground_truth_answer
    AND
        predicted_rationale == ground_truth_rationale
    
    This is the PRIMARY evaluation metric for VCR.
    
    IMPORTANT: The rationale prediction here must use the PREDICTED answer
    (not the ground-truth answer). Using ground-truth answer would overestimate.
    """
    assert len(answer_preds) == len(answer_labels) == len(rationale_preds) == len(rationale_labels)
    if len(answer_preds) == 0:
        return 0.0
    correct = sum(
        ap == al and rp == rl
        for ap, al, rp, rl in zip(answer_preds, answer_labels, rationale_preds, rationale_labels)
    )
    return correct / len(answer_preds)


def compute_all_metrics(answer_preds, answer_labels, rationale_preds, rationale_labels,
                        rationale_preds_gt_answer=None, rationale_labels_for_gt=None) -> dict:
    """Compute all VCR metrics.
    
    Returns dict with: q_to_a, qa_to_r, q_to_ar, and optionally qa_to_r_gt_answer
    """
    result = {
        'q_to_a': compute_q_to_a(answer_preds, answer_labels),
        'q_to_ar': compute_q_to_ar(answer_preds, answer_labels, rationale_preds, rationale_labels),
    }
    
    # QA→R with GT answer (if provided separately)
    if rationale_preds_gt_answer is not None and rationale_labels_for_gt is not None:
        result['qa_to_r'] = compute_qa_to_r(rationale_preds_gt_answer, rationale_labels_for_gt)
    
    return result
