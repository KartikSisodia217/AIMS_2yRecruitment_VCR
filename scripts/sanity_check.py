"""End-to-end CPU sanity check for the CACR-SP pipeline.

Runs the complete pipeline on synthetic data with a mock VLM:
1. Create synthetic VCR samples
2. Initialize mock VLM backbone
3. Build all model components
4. Run Stage 1: answer scoring
5. Run Stage 2: rationale scoring via CACR
6. Compute contrastive loss
7. Compute shortcut penalty
8. Compute total loss
9. Run backward pass
10. Run optimizer step
11. Compute evaluation metrics
12. Report results

This must work entirely on CPU with no real VLM or real data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.debug_dataset import DebugVCRDataset, create_debug_vcr_sample
from data.collator import VCRCollator
from models.vlm_backbone import MockVLMBackbone
from models.answer_scorer import MockAnswerScorer
from models.projection import ProjectionHead
from models.rationale_encoder import MockRationaleEncoder
from models.similarity import CosineSimilarity
from models.cacr_sp import CACRSPModel
from losses.contrastive import ContrastiveLoss
from losses.shortcut import ShortcutPenalty
from losses.total_loss import TotalLoss
from evaluation.metrics import compute_q_to_a, compute_qa_to_r, compute_q_to_ar
from utils.seed import set_seed
from utils.param_count import count_parameters, print_parameter_summary

def main():
    print("="*60)
    print("CACR-SP VCR Pipeline - CPU Sanity Check")
    print("="*60)
    
    set_seed(42)
    
    # Step 1: Create synthetic data
    print("\n[1/12] Creating synthetic VCR data...")
    dataset = DebugVCRDataset(num_samples=5, create_images=True)
    sample = dataset[0]
    print(f"  Created {len(dataset)} samples")
    print(f"  Sample 0: Q='{sample.question[:50]}...'")
    print(f"  Answer choices: {len(sample.answer_choices)}")
    print(f"  Rationale choices: {len(sample.rationale_choices)}")
    print(f"  Labels: A={sample.answer_label}, R={sample.rationale_label}")
    assert sample.is_valid, f"Sample validation failed: {sample.validate()}"
    
    # Step 2: Test collator
    print("\n[2/12] Testing batch collation...")
    collator = VCRCollator(load_images=False)
    batch = collator([dataset[i] for i in range(3)])
    print(f"  Batch size: {len(batch.sample_ids)}")
    print(f"  Answer labels shape: {batch.answer_labels.shape}")
    
    # Step 3: Initialize mock VLM
    HIDDEN_DIM = 256
    EMBEDDING_DIM = 128
    print(f"\n[3/12] Initializing MockVLM (hidden_dim={HIDDEN_DIM})...")
    vlm = MockVLMBackbone(hidden_dim=HIDDEN_DIM)
    print(f"  Hidden dim: {vlm.get_hidden_dim()}")
    print(f"  Device: {vlm.device}")
    
    # Step 4: Build model components
    print("\n[4/12] Building model components...")
    answer_scorer = MockAnswerScorer(default_prediction=0)
    projection = ProjectionHead(
        input_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        intermediate_dim=HIDDEN_DIM,
    )
    rationale_encoder = MockRationaleEncoder(embedding_dim=EMBEDDING_DIM)
    similarity_fn = CosineSimilarity(temperature=0.07)
    
    # Step 5: Assemble CACR-SP model
    print("\n[5/12] Assembling CACRSPModel...")
    model = CACRSPModel(
        vlm=vlm,
        answer_scorer=answer_scorer,
        projection=projection,
        rationale_encoder=rationale_encoder,
        similarity_fn=similarity_fn,
    )
    print_parameter_summary(model, "CACRSPModel")
    
    # Step 6: Build loss functions
    print("\n[6/12] Building loss functions...")
    contrastive_loss = ContrastiveLoss(loss_type='infonce', temperature=0.07)
    shortcut_penalty = ShortcutPenalty(lambda_sp=0.1, margin=0.25)
    total_loss_fn = TotalLoss(
        contrastive_loss=contrastive_loss,
        shortcut_penalty=shortcut_penalty,
        lambda_sp=0.1,
    )
    
    # Step 7: Forward pass — Stage 1 (answer scoring)
    print("\n[7/12] Stage 1: Answer scoring...")
    answer_pred, answer_scores = model.predict_answer(
        None, sample.question, sample.answer_choices
    )
    print(f"  Answer scores: {answer_scores}")
    print(f"  Predicted answer: {answer_pred} (GT: {sample.answer_label})")
    
    # Step 8: Forward pass — Stage 2 (rationale scoring via CACR)
    print("\n[8/12] Stage 2: Rationale scoring...")
    selected_answer = sample.answer_choices[sample.answer_label]  # Use GT for debugging
    rat_scores, context_emb, rat_embs = model.score_rationales(
        None, sample.question, selected_answer, sample.rationale_choices
    )
    print(f"  Context embedding shape: {context_emb.shape}")
    print(f"  Rationale embeddings shape: {rat_embs.shape}")
    print(f"  Similarity scores: {rat_scores}")
    rationale_pred = rat_scores.argmax().item()
    print(f"  Predicted rationale: {rationale_pred} (GT: {sample.rationale_label})")
    
    # Step 9: Compute losses
    print("\n[9/12] Computing losses...")
    label_tensor = torch.tensor(sample.rationale_label)
    
    # Blind branch
    blind_scores_list = []
    for r in sample.rationale_choices:
        blind_repr = model.forward_blind(sample.question, r)
        blind_emb = projection(blind_repr.unsqueeze(0)).squeeze(0)
        blind_scores_list.append(similarity_fn.compute(blind_emb, rat_embs).mean())
    blind_scores = torch.stack(blind_scores_list)
    
    loss_dict = total_loss_fn(
        rationale_scores=rat_scores.unsqueeze(0),
        rationale_label=label_tensor.unsqueeze(0),
        blind_scores=blind_scores.unsqueeze(0),
    )
    print(f"  Contrastive loss: {loss_dict['contrastive'].item():.4f}")
    print(f"  Shortcut penalty: {loss_dict['shortcut'].item():.4f}")
    print(f"  Total loss: {loss_dict['total'].item():.4f}")
    
    # Step 10: Backward pass
    print("\n[10/12] Backward pass...")
    loss_dict['total'].backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {grad_count}")
    
    # Step 11: Optimizer step
    print("\n[11/12] Optimizer step...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    optimizer.step()
    optimizer.zero_grad()
    print("  Optimizer step completed successfully")
    
    # Step 12: Evaluation metrics
    print("\n[12/12] Computing evaluation metrics...")
    # Run predictions on all debug samples
    answer_preds = []
    answer_labels = []
    rationale_preds = []
    rationale_labels = []
    
    for i in range(len(dataset)):
        s = dataset[i]
        result = model.predict(
            None, s.question, s.answer_choices, s.rationale_choices
        )
        answer_preds.append(result['predicted_answer'])
        answer_labels.append(s.answer_label)
        rationale_preds.append(result['predicted_rationale'])
        rationale_labels.append(s.rationale_label)
    
    q_to_a = compute_q_to_a(answer_preds, answer_labels)
    qa_to_r = compute_qa_to_r(rationale_preds, rationale_labels)
    q_to_ar = compute_q_to_ar(answer_preds, answer_labels, rationale_preds, rationale_labels)
    
    print(f"  Q->A accuracy:  {q_to_a:.2%}")
    print(f"  QA->R accuracy: {qa_to_r:.2%}")
    print(f"  Q->AR accuracy: {q_to_ar:.2%}")
    print(f"  (Using mock model - metrics are not meaningful, only verifying correctness)")
    
    # Summary
    print("\n" + "="*60)
    print("[PASS] ALL SANITY CHECKS PASSED")
    print("="*60)
    print(f"\nPipeline verified:")
    print(f"  synthetic VCR sample")
    print(f"  -> mock VLM")
    print(f"  -> answer scoring (Stage 1)")
    print(f"  -> context representation")
    print(f"  -> projection head")
    print(f"  -> rationale embeddings")
    print(f"  -> cosine similarity")
    print(f"  -> rationale prediction (Stage 2)")
    print(f"  -> contrastive loss")
    print(f"  -> shortcut penalty (placeholder)")
    print(f"  -> total loss")
    print(f"  -> backward pass")
    print(f"  -> optimizer step")
    print(f"  -> evaluation metrics")
    print(f"\nReady for real VLM and real VCR data.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
