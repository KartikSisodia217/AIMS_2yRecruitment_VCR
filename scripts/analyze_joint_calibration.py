"""
CUDA-only inference analysis script for 16-way Joint CACR calibration.

This script:
1. Loads the trained checkpoint (no retraining)
2. Collects raw answer logits and rationale scores on the validation set
3. Reports score distribution statistics
4. Performs an inference-only alpha/beta weighting sweep
5. Reports Q->AR (and Q->A, QA->R) for each weighting

MUST be run on CUDA. Will refuse to run on CPU.

Usage:
    python scripts/analyze_joint_calibration.py \
        --checkpoint checkpoints/cacr_sp/best_model.pt \
        --data_dir data/vcr \
        --image_dir /path/to/vcr1images_clean \
        --max_val_samples 512 \
        --batch_size 4
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image


def collect_scores(model, dataloader, device):
    """Run one evaluation pass collecting raw scores for all samples."""
    model.eval()
    
    all_ans_logits = []
    all_rat_scores = []  # raw cosine similarities [B, 4, 4]
    all_ans_labels = []
    all_rat_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = [load_image(p) for p in batch["image_path"]]
            questions = batch["question"]
            answer_choices = batch["answer_choices"]
            ans_labels = batch["answer_label"].to(device)
            rationale_choices = batch["rationale_choices"]
            rat_labels = batch["rationale_label"].to(device)
            
            image_embs = model.encode_images(images)
            
            # Get answer logits
            ans_logits = model.forward_answer(
                images, questions, answer_choices, image_embs=image_embs
            )
            
            # Get raw rationale scores (cosine similarities, NOT temperature-scaled)
            result = model.forward_joint_rationale(
                images, questions, answer_choices, rationale_choices, image_embs=image_embs
            )
            joint_rat_scores = result["joint_rationale_scores"]  # [B, 4, 4]
            
            all_ans_logits.append(ans_logits.cpu())
            all_rat_scores.append(joint_rat_scores.cpu())
            all_ans_labels.append(ans_labels.cpu())
            all_rat_labels.append(rat_labels.cpu())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Collected {(batch_idx + 1) * ans_logits.shape[0]} samples...")
    
    return {
        "ans_logits": torch.cat(all_ans_logits, dim=0),      # [N, 4]
        "rat_scores": torch.cat(all_rat_scores, dim=0),       # [N, 4, 4]
        "ans_labels": torch.cat(all_ans_labels, dim=0),       # [N]
        "rat_labels": torch.cat(all_rat_labels, dim=0),       # [N]
    }


def print_distribution(name, tensor):
    """Print distribution statistics for a tensor."""
    vals = tensor.float()
    print(f"\n  {name}:")
    print(f"    mean:   {vals.mean().item():+.4f}")
    print(f"    std:    {vals.std().item():.4f}")
    print(f"    min:    {vals.min().item():+.4f}")
    print(f"    max:    {vals.max().item():+.4f}")
    print(f"    median: {vals.median().item():+.4f}")
    # Percentiles
    sorted_vals = vals.flatten().sort().values
    n = len(sorted_vals)
    p5 = sorted_vals[int(0.05 * n)].item()
    p25 = sorted_vals[int(0.25 * n)].item()
    p75 = sorted_vals[int(0.75 * n)].item()
    p95 = sorted_vals[int(0.95 * n)].item()
    print(f"    p5:     {p5:+.4f}")
    print(f"    p25:    {p25:+.4f}")
    print(f"    p75:    {p75:+.4f}")
    print(f"    p95:    {p95:+.4f}")


def evaluate_with_weights(ans_logits, rat_scores, ans_labels, rat_labels, alpha, beta, temperature):
    """
    Evaluate Q->AR, Q->A, QA->R with given alpha/beta weights.
    
    Joint score = alpha * ans_logits[a] + beta * rat_scores[a,r] / temperature
    """
    N = ans_logits.shape[0]
    
    # Scale scores
    scaled_rat = rat_scores / temperature  # [N, 4, 4]
    joint_scores = alpha * ans_logits.unsqueeze(2) + beta * scaled_rat  # [N, 4, 4]
    
    # Q->AR: argmax over 16 combinations
    joint_flat = joint_scores.view(N, 16)
    joint_preds = joint_flat.argmax(dim=-1)
    pred_a = joint_preds // 4
    pred_r = joint_preds % 4
    
    match_a_joint = (pred_a == ans_labels)
    match_r_joint = (pred_r == rat_labels)
    match_ar = match_a_joint & match_r_joint
    
    acc_ar = match_ar.float().mean().item()
    
    # Q->A from joint: which answer does the joint objective pick?
    acc_a_joint = match_a_joint.float().mean().item()
    
    # Q->A standalone (independent of alpha/beta, just for reference)
    ans_preds = ans_logits.argmax(dim=-1)
    acc_a_standalone = (ans_preds == ans_labels).float().mean().item()
    
    # QA->R (teacher-forced): given correct answer, pick rationale
    tf_rat_scores = scaled_rat[torch.arange(N), ans_labels, :]  # [N, 4]
    tf_rat_preds = tf_rat_scores.argmax(dim=-1)
    acc_r_tf = (tf_rat_preds == rat_labels).float().mean().item()
    
    # QA->R (predicted answer): given predicted answer from joint, pick rationale
    pred_a_for_rat = pred_a
    pred_rat_scores = scaled_rat[torch.arange(N), pred_a_for_rat, :]
    pred_rat_preds = pred_rat_scores.argmax(dim=-1)
    acc_r_pred = (pred_rat_preds == rat_labels).float().mean().item()
    
    return {
        "acc_ar": acc_ar,
        "acc_a_joint": acc_a_joint,
        "acc_a_standalone": acc_a_standalone,
        "acc_r_tf": acc_r_tf,
        "acc_r_pred": acc_r_pred,
    }


def dominance_analysis(ans_logits, rat_scores, ans_labels, rat_labels, temperature):
    """Analyze which branch dominates joint decisions."""
    N = ans_logits.shape[0]
    
    scaled_rat = rat_scores / temperature  # [N, 4, 4]
    
    # For each sample, find:
    # 1. The answer contribution range (max - min across 4 answers)
    # 2. The rationale contribution range (max - min across 4x4 rationale scores)
    
    ans_range = ans_logits.max(dim=1).values - ans_logits.min(dim=1).values  # [N]
    rat_range = scaled_rat.view(N, 16).max(dim=1).values - scaled_rat.view(N, 16).min(dim=1).values  # [N]
    
    # Which has larger dynamic range per sample?
    ans_dominates = (ans_range > rat_range).float().mean().item()
    rat_dominates = (rat_range > ans_range).float().mean().item()
    
    # For the joint argmax decision, check if changing ONLY the answer term
    # vs changing ONLY the rationale term would change the decision
    
    # Method: for each sample, compute joint argmax.
    # Then compute argmax with ans_logits zeroed (rationale-only) and rat_scores zeroed (answer-only)
    joint_scores = ans_logits.unsqueeze(2) + scaled_rat  # [N, 4, 4]
    joint_preds = joint_scores.view(N, 16).argmax(dim=-1)
    
    # Rationale-only prediction
    rat_only_preds = scaled_rat.view(N, 16).argmax(dim=-1)
    
    # Answer-only prediction (pick best answer, then best rationale for that answer)
    ans_only_preds_a = ans_logits.argmax(dim=-1)
    ans_only_rat_scores = scaled_rat[torch.arange(N), ans_only_preds_a, :]  # [N, 4]
    ans_only_preds_r = ans_only_rat_scores.argmax(dim=-1)
    ans_only_preds = ans_only_preds_a * 4 + ans_only_preds_r
    
    # How often does joint match rationale-only?
    joint_matches_rat_only = (joint_preds == rat_only_preds).float().mean().item()
    # How often does joint match the answer-first-then-rationale strategy?
    joint_matches_ans_first = (joint_preds == ans_only_preds).float().mean().item()
    
    print(f"\n  === Dominance Analysis ===")
    print(f"  Answer logit dynamic range:   mean={ans_range.mean():.4f}, std={ans_range.std():.4f}")
    print(f"  Rationale score dynamic range: mean={rat_range.mean():.4f}, std={rat_range.std():.4f}")
    print(f"  Ratio (rat/ans range):         {(rat_range / (ans_range + 1e-8)).mean():.2f}x")
    print(f"")
    print(f"  Answer dominates (per-sample): {ans_dominates*100:.1f}%")
    print(f"  Rationale dominates:           {rat_dominates*100:.1f}%")
    print(f"")
    print(f"  Joint == rationale-only:       {joint_matches_rat_only*100:.1f}%")
    print(f"  Joint == answer-first-then-R:  {joint_matches_ans_first*100:.1f}%")
    
    return {
        "ans_range_mean": ans_range.mean().item(),
        "rat_range_mean": rat_range.mean().item(),
        "ratio": (rat_range / (ans_range + 1e-8)).mean().item(),
        "ans_dominates_pct": ans_dominates * 100,
        "rat_dominates_pct": rat_dominates * 100,
        "joint_eq_rat_only_pct": joint_matches_rat_only * 100,
        "joint_eq_ans_first_pct": joint_matches_ans_first * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Joint CACR Calibration Analysis (CUDA only)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cacr_sp/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data/vcr")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing extracted images (e.g. vcr1images_clean)")
    parser.add_argument("--max_val_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # === CUDA CHECK ===
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script MUST run on a CUDA device.")
        print("Do NOT fall back to CPU. Exiting.")
        sys.exit(1)
    
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    
    set_seed(args.seed)
    
    # Load validation data
    print("\nLoading validation dataset...")
    val_dataset = VCRDataset(split="val", data_dir=args.data_dir, image_dir=args.image_dir)
    if args.max_val_samples > 0:
        indices = list(range(min(args.max_val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
    print(f"  Val samples: {len(val_dataset)}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=vcr_collate_fn
    )
    
    # Load model
    print("\nLoading model...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device="cuda")
    model = CACRSPVCRModel(
        vlm=vlm, scorer_dropout=0.1,
        embedding_dim=args.embedding_dim, temperature=args.temperature
    )
    model.to(device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    if "val_metrics" in checkpoint:
        print(f"  Checkpoint val metrics: {checkpoint['val_metrics']}")
    
    # ==========================================
    # PHASE 1: Collect raw scores
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 1: Collecting raw scores")
    print("="*60)
    
    data = collect_scores(model, val_loader, device)
    
    ans_logits = data["ans_logits"]      # [N, 4]
    rat_scores = data["rat_scores"]      # [N, 4, 4]
    ans_labels = data["ans_labels"]      # [N]
    rat_labels = data["rat_labels"]      # [N]
    
    N = ans_logits.shape[0]
    print(f"\n  Total samples collected: {N}")
    
    # ==========================================
    # PHASE 2: Score distribution analysis
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 2: Score Distribution Analysis")
    print("="*60)
    
    # Answer logits
    print_distribution("ANSWER LOGITS (raw)", ans_logits)
    
    # Rationale scores (raw cosine similarities)
    print_distribution("RATIONALE SCORES (raw cosine sim)", rat_scores)
    
    # Rationale scores after temperature
    scaled_rat = rat_scores / args.temperature
    print_distribution("RATIONALE SCORES (after /temperature)", scaled_rat)
    
    # Joint scores (current alpha=1, beta=1)
    joint = ans_logits.unsqueeze(2) + scaled_rat
    print_distribution("JOINT SCORES (ans + rat/tau)", joint)
    
    # Contribution analysis
    ans_contribution = ans_logits.unsqueeze(2).expand_as(joint).abs()
    rat_contribution = scaled_rat.abs()
    
    print(f"\n  === Contribution Magnitude ===")
    print(f"  Avg |answer contribution|:    {ans_contribution.mean().item():.4f}")
    print(f"  Avg |rationale contribution|: {rat_contribution.mean().item():.4f}")
    print(f"  Ratio (rat/ans):              {rat_contribution.mean().item() / (ans_contribution.mean().item() + 1e-8):.2f}x")
    
    # ==========================================
    # PHASE 3: Dominance analysis
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 3: Dominance Analysis")
    print("="*60)
    
    dominance_analysis(ans_logits, rat_scores, ans_labels, rat_labels, args.temperature)
    
    # ==========================================
    # PHASE 4: Inference-only alpha/beta sweep
    # ==========================================
    print("\n" + "="*60)
    print("PHASE 4: Inference-Only alpha/beta Weighting Sweep")
    print("="*60)
    
    sweep_configs = [
        # (alpha, beta, description)
        (1.0,  1.0,  "CURRENT (baseline)"),
        (1.0,  0.5,  "halve rationale"),
        (1.0,  0.75, "reduce rationale 25%"),
        (1.0,  1.25, "boost rationale 25%"),
        (1.0,  1.5,  "boost rationale 50%"),
        (0.75, 1.0,  "reduce answer 25%"),
        (1.25, 1.0,  "boost answer 25%"),
        (1.5,  1.0,  "boost answer 50%"),
        (2.0,  1.0,  "double answer"),
        (2.0,  0.75, "double answer + reduce rationale"),
        (3.0,  1.0,  "triple answer"),
        # Additional fine-grained sweep around promising region
        (1.5,  0.75, "1.5x answer + 0.75x rationale"),
        (1.25, 0.75, "1.25x answer + 0.75x rationale"),
        (1.75, 1.0,  "1.75x answer"),
        (2.5,  1.0,  "2.5x answer"),
    ]
    
    print(f"\n  {'a':>6}  {'b':>6}  {'Q->AR':>8}  {'Q->A(J)':>8}  {'Q->A(S)':>8}  {'QA->R(TF)':>10}  {'QA->R(P)':>9}  Description")
    print(f"  {'---':>6}  {'---':>6}  {'------':>8}  {'------':>8}  {'------':>8}  {'--------':>10}  {'-------':>9}  -----------")
    
    best_ar = 0.0
    best_config = None
    results = []
    
    for alpha, beta, desc in sweep_configs:
        metrics = evaluate_with_weights(
            ans_logits, rat_scores, ans_labels, rat_labels,
            alpha, beta, args.temperature
        )
        
        marker = " << BEST" if metrics["acc_ar"] > best_ar else ""
        if metrics["acc_ar"] > best_ar:
            best_ar = metrics["acc_ar"]
            best_config = (alpha, beta, desc)
        
        print(f"  {alpha:6.2f}  {beta:6.2f}  {metrics['acc_ar']*100:7.2f}%  {metrics['acc_a_joint']*100:7.2f}%  {metrics['acc_a_standalone']*100:7.2f}%  {metrics['acc_r_tf']*100:9.2f}%  {metrics['acc_r_pred']*100:8.2f}%  {desc}{marker}")
        
        results.append({
            "alpha": alpha, "beta": beta, "desc": desc,
            **metrics
        })
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    current = [r for r in results if r["alpha"] == 1.0 and r["beta"] == 1.0][0]
    
    print(f"\n  Current (a=1, b=1):")
    print(f"    Q->AR:  {current['acc_ar']*100:.2f}%")
    print(f"    Q->A:   {current['acc_a_standalone']*100:.2f}%")
    
    if best_config:
        alpha, beta, desc = best_config
        best_result = [r for r in results if r["alpha"] == alpha and r["beta"] == beta][0]
        
        print(f"\n  Best (a={alpha}, b={beta} -- {desc}):")
        print(f"    Q->AR:  {best_result['acc_ar']*100:.2f}%")
        print(f"    Q->A:   {best_result['acc_a_standalone']*100:.2f}%")
        print(f"    Delta Q->AR: {(best_result['acc_ar'] - current['acc_ar'])*100:+.2f}pp")
        
        if best_result['acc_ar'] > current['acc_ar']:
            print(f"\n  >>> RECOMMENDATION: Use a={alpha}, b={beta} for controlled training experiment")
            print(f"  >>> This is an inference-only result. A training experiment is needed to confirm.")
        else:
            print(f"\n  >>> No improvement found from reweighting. Investigate other directions.")
    
    print(f"\n  All results saved. Proceed to next analysis step based on findings.")


if __name__ == "__main__":
    main()
