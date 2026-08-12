"""
Numerical Equivalence Test: Old (repeated-image joint forward) vs New (cached image embedding reuse).

This test verifies that the optimized image embedding reuse produces identical logits
to the original approach where images were duplicated 4x and processed through the joint forward.
"""

import os
import torch
import torch.nn as nn
from PIL import Image

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.model import BaselineVCRModel, BaselineScorer
from src.utils import load_image, set_seed


def build_old_style_logits(vlm, scorer, images, texts_flat, B):
    """
    Replicates the OLD behavior: duplicate images 4x, call joint forward, score.
    This is the reference implementation we compare against.
    """
    images_flat = [img for img in images for _ in range(4)]
    
    # Joint forward (old style — encodes each image 4 times)
    vlm_outputs = vlm(images=images_flat, texts=texts_flat)
    img_emb = vlm_outputs["image_embeds"]  # [B*4, 768]
    text_emb = vlm_outputs["text_embeds"]  # [B*4, 768]
    
    # Score using the scorer
    scores = scorer(img_emb, text_emb)  # [B*4, 1]
    logits = scores.view(B, 4)
    return logits


def build_new_style_logits(vlm, scorer, images, texts_flat, B):
    """
    The NEW behavior: encode images once, encode texts separately, expand, score.
    """
    # Encode images once
    image_embs = vlm.encode_image(images)  # [B, 768]
    
    # Encode texts
    text_embs = vlm.encode_text(texts_flat)  # [B*4, 768]
    
    # Expand image embeddings
    img_embs_expanded = image_embs.unsqueeze(1).expand(B, 4, -1).reshape(B * 4, -1)
    
    # Score
    scores = scorer(img_embs_expanded, text_embs)
    logits = scores.view(B, 4)
    return logits


def main():
    set_seed(42)
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")
    
    # Load dataset
    dataset = VCRDataset(split='train', data_dir='data/vcr')
    zip_path = os.path.join("data", "vcr", "vcr1images.zip")
    
    # Load VLM
    print("Loading SigLIP2...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device_str)
    
    # Create a scorer with fixed weights
    set_seed(42)
    scorer = BaselineScorer(dropout=0.0)  # No dropout for determinism
    scorer.to(device)
    scorer.eval()
    
    # Test on 2 samples (B=2)
    num_samples = 2
    images = []
    questions = []
    answer_choices_list = []
    rationale_choices_list = []
    
    for i in range(num_samples):
        sample = dataset[i]
        img = load_image(sample["image_path"], zip_path)
        images.append(img)
        questions.append(sample["question"])
        answer_choices_list.append(sample["answer_choices"])
        rationale_choices_list.append(sample["rationale_choices"])
    
    B = len(images)
    
    print(f"\nTesting with B={B} samples...")
    
    # ============================================================
    # TEST 1: Answer logits equivalence
    # ============================================================
    print("\n=== Answer Logits Equivalence ===")
    
    ans_texts_flat = []
    for q, a_list in zip(questions, answer_choices_list):
        for a in a_list:
            ans_texts_flat.append(f"Question: {q} Answer: {a}")
    
    with torch.no_grad():
        old_ans_logits = build_old_style_logits(vlm, scorer, images, ans_texts_flat, B)
        new_ans_logits = build_new_style_logits(vlm, scorer, images, ans_texts_flat, B)
    
    ans_diff = (old_ans_logits - new_ans_logits).abs()
    print(f"Old answer logits:\n{old_ans_logits}")
    print(f"New answer logits:\n{new_ans_logits}")
    print(f"Max absolute difference: {ans_diff.max().item():.2e}")
    print(f"Mean absolute difference: {ans_diff.mean().item():.2e}")
    print(f"Exact match: {torch.equal(old_ans_logits, new_ans_logits)}")
    
    # ============================================================
    # TEST 2: Rationale logits equivalence
    # ============================================================
    print("\n=== Rationale Logits Equivalence ===")
    
    selected_answers = [answer_choices_list[i][0] for i in range(B)]  # Use first answer for consistency
    
    rat_texts_flat = []
    for q, ans, r_list in zip(questions, selected_answers, rationale_choices_list):
        for r in r_list:
            rat_texts_flat.append(f"Question: {q} Answer: {ans} Rationale: {r}")
    
    with torch.no_grad():
        old_rat_logits = build_old_style_logits(vlm, scorer, images, rat_texts_flat, B)
        new_rat_logits = build_new_style_logits(vlm, scorer, images, rat_texts_flat, B)
    
    rat_diff = (old_rat_logits - new_rat_logits).abs()
    print(f"Old rationale logits:\n{old_rat_logits}")
    print(f"New rationale logits:\n{new_rat_logits}")
    print(f"Max absolute difference: {rat_diff.max().item():.2e}")
    print(f"Mean absolute difference: {rat_diff.mean().item():.2e}")
    print(f"Exact match: {torch.equal(old_rat_logits, new_rat_logits)}")
    
    # ============================================================
    # TEST 3: End-to-end model equivalence
    # ============================================================
    print("\n=== End-to-End Model Forward Equivalence ===")
    
    # Build full model
    set_seed(42)
    model = BaselineVCRModel(vlm=vlm, scorer_dropout=0.0)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        # New path (with image embedding reuse)
        image_embs = model.encode_images(images)
        new_model_ans = model.forward_answer(images, questions, answer_choices_list, image_embs=image_embs)
        new_model_rat = model.forward_rationale(
            images, questions, selected_answers, rationale_choices_list, image_embs=image_embs
        )
        
        # Old path (without image embedding reuse — falls back to encode inside)
        old_model_ans = model.forward_answer(images, questions, answer_choices_list)
        old_model_rat = model.forward_rationale(
            images, questions, selected_answers, rationale_choices_list
        )
    
    ans_model_diff = (old_model_ans - new_model_ans).abs()
    rat_model_diff = (old_model_rat - new_model_rat).abs()
    
    print(f"Answer max diff: {ans_model_diff.max().item():.2e}")
    print(f"Answer mean diff: {ans_model_diff.mean().item():.2e}")
    print(f"Answer exact match: {torch.equal(old_model_ans, new_model_ans)}")
    print(f"Rationale max diff: {rat_model_diff.max().item():.2e}")
    print(f"Rationale mean diff: {rat_model_diff.mean().item():.2e}")
    print(f"Rationale exact match: {torch.equal(old_model_rat, new_model_rat)}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    overall_max = max(ans_diff.max().item(), rat_diff.max().item(), 
                      ans_model_diff.max().item(), rat_model_diff.max().item())
    overall_mean = (ans_diff.mean().item() + rat_diff.mean().item() + 
                    ans_model_diff.mean().item() + rat_model_diff.mean().item()) / 4
    
    print(f"\n{'='*60}")
    print(f"OVERALL MAX ABSOLUTE DIFFERENCE: {overall_max:.2e}")
    print(f"OVERALL MEAN ABSOLUTE DIFFERENCE: {overall_mean:.2e}")
    
    if overall_max < 1e-6:
        print("PASS — Numerical equivalence confirmed.")
    elif overall_max < 1e-4:
        print("WARNING — Small numerical differences (floating-point expected).")
    else:
        print("FAIL — Significant differences detected!")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
