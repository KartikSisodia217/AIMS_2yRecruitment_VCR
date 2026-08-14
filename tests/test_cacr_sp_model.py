"""
Unit test for CACR-SP model: shape verification, gradient flow, and loss integration.
Tests the full forward pass with synthetic data (no VLM loading).
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import BaselineScorer
from models.projection import ProjectionHead
from losses.contrastive import ContrastiveLoss
from losses.shortcut import ShortcutPenalty


class MockSigLIP2:
    """Minimal mock of SigLIP2Wrapper for CPU testing."""
    def __init__(self, embed_dim=768):
        self.embed_dim = embed_dim
        self._dummy = nn.Linear(1, 1)  # for .parameters()
        # Freeze
        for p in self._dummy.parameters():
            p.requires_grad = False
    
    def parameters(self):
        return self._dummy.parameters()
    
    def encode_image(self, images):
        B = len(images)
        embs = torch.randn(B, self.embed_dim)
        return embs / embs.norm(p=2, dim=-1, keepdim=True)
    
    def encode_text(self, texts):
        B = len(texts)
        embs = torch.randn(B, self.embed_dim)
        return embs / embs.norm(p=2, dim=-1, keepdim=True)


def test_cacr_sp_model_shapes():
    """Test that all tensor shapes are correct throughout the forward pass."""
    from src.cacr_sp_model import CACRSPVCRModel
    
    mock_vlm = MockSigLIP2()
    model = CACRSPVCRModel(vlm=mock_vlm, scorer_dropout=0.0, embedding_dim=512)
    model.eval()
    
    B = 2
    images = [None] * B  # Mock doesn't use real images
    questions = ["Why is person looking at dog?", "What is happening?"]
    answer_choices = [
        ["answer A0", "answer A1", "answer A2", "answer A3"],
        ["answer B0", "answer B1", "answer B2", "answer B3"],
    ]
    rationale_choices = [
        ["rationale A0", "rationale A1", "rationale A2", "rationale A3"],
        ["rationale B0", "rationale B1", "rationale B2", "rationale B3"],
    ]
    selected_answers = ["answer A2", "answer B1"]
    
    # Test encode_images
    image_embs = model.encode_images(images)
    assert image_embs.shape == (B, 768), f"Expected [B, 768], got {image_embs.shape}"
    
    # Test forward_answer
    ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
    assert ans_logits.shape == (B, 4), f"Expected [B, 4], got {ans_logits.shape}"
    
    # Test forward_rationale
    result = model.forward_rationale(images, questions, selected_answers, rationale_choices, image_embs=image_embs)
    
    assert 'rationale_scores' in result
    assert 'blind_scores' in result
    assert 'context_emb' in result
    assert 'rationale_embs' in result
    
    assert result['rationale_scores'].shape == (B, 4), f"Expected [B, 4], got {result['rationale_scores'].shape}"
    assert result['blind_scores'].shape == (B, 4), f"Expected [B, 4], got {result['blind_scores'].shape}"
    assert result['context_emb'].shape == (B, 512), f"Expected [B, 512], got {result['context_emb'].shape}"
    assert result['rationale_embs'].shape == (B, 4, 512), f"Expected [B, 4, 512], got {result['rationale_embs'].shape}"
    
    # Test that rationale_scores are cosine similarities (bounded [-1, 1])
    assert result['rationale_scores'].min() >= -1.01, "Scores below -1"
    assert result['rationale_scores'].max() <= 1.01, "Scores above 1"
    
    print("✓ All shapes correct")


def test_cacr_sp_gradient_flow():
    """Test that gradients flow to all trainable parameters."""
    from src.cacr_sp_model import CACRSPVCRModel
    
    mock_vlm = MockSigLIP2()
    model = CACRSPVCRModel(vlm=mock_vlm, scorer_dropout=0.0, embedding_dim=256)
    model.train()
    
    B = 2
    images = [None] * B
    questions = ["Q1", "Q2"]
    answer_choices = [["a0","a1","a2","a3"], ["b0","b1","b2","b3"]]
    rationale_choices = [["r0","r1","r2","r3"], ["s0","s1","s2","s3"]]
    selected_answers = ["a2", "b1"]
    
    ans_labels = torch.tensor([2, 1])
    rat_labels = torch.tensor([0, 3])
    
    image_embs = model.encode_images(images)
    
    # Answer loss
    ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
    ans_loss = nn.CrossEntropyLoss()(ans_logits, ans_labels)
    
    # Rationale loss
    result = model.forward_rationale(images, questions, selected_answers, rationale_choices, image_embs=image_embs)
    contrastive_loss = ContrastiveLoss(loss_type='infonce', temperature=0.07)
    contr_loss = contrastive_loss(result['rationale_scores'], rat_labels)
    
    # Shortcut penalty
    sp = ShortcutPenalty(margin=0.25, formulation='confidence_penalty')
    sp_loss = sp(result['blind_scores'], rat_labels)
    
    # Total loss
    total_loss = ans_loss + contr_loss + 0.1 * sp_loss
    total_loss.backward()
    
    # Check gradients exist for all trainable components
    components = {
        'answer_scorer': model.answer_scorer,
        'context_projection': model.context_projection,
        'rationale_projection': model.rationale_projection,
        'blind_projection': model.blind_projection,
    }
    
    for name, component in components.items():
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in component.parameters() if p.requires_grad)
        assert has_grad, f"No gradients in {name}"
        print(f"  ✓ Gradients flow to {name}")
    
    print("✓ All gradient flows correct")


def test_cacr_sp_trainable_params():
    """Test parameter counts and frozen/trainable status."""
    from src.cacr_sp_model import CACRSPVCRModel
    
    mock_vlm = MockSigLIP2()
    model = CACRSPVCRModel(vlm=mock_vlm, scorer_dropout=0.0, embedding_dim=512)
    
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Frozen parameters: {frozen:,}")
    print(f"  Trainable parameters: {trainable:,}")
    
    # VLM should be frozen
    for p in model.vlm.parameters():
        assert not p.requires_grad, "VLM should be frozen"
    
    # Answer scorer should be trainable
    for p in model.answer_scorer.parameters():
        assert p.requires_grad, "Answer scorer should be trainable"
    
    # Projection heads should be trainable
    for p in model.context_projection.parameters():
        assert p.requires_grad, "Context projection should be trainable"
    for p in model.rationale_projection.parameters():
        assert p.requires_grad, "Rationale projection should be trainable"
    for p in model.blind_projection.parameters():
        assert p.requires_grad, "Blind projection should be trainable"
    
    assert trainable > 0, "No trainable parameters"
    print("✓ Parameter status correct")


def test_loss_integration():
    """Test that all loss components work together with the model output."""
    from src.cacr_sp_model import CACRSPVCRModel
    
    mock_vlm = MockSigLIP2()
    model = CACRSPVCRModel(vlm=mock_vlm, embedding_dim=256)
    model.train()
    
    B = 3
    images = [None] * B
    questions = ["Q1", "Q2", "Q3"]
    selected_answers = ["a", "b", "c"]
    rationale_choices = [["r0","r1","r2","r3"]] * B
    rat_labels = torch.tensor([0, 1, 2])
    
    image_embs = model.encode_images(images)
    result = model.forward_rationale(images, questions, selected_answers, rationale_choices, image_embs=image_embs)
    
    # InfoNCE loss
    infonce = ContrastiveLoss(loss_type='infonce', temperature=0.07)
    loss_infonce = infonce(result['rationale_scores'], rat_labels)
    assert loss_infonce.item() > 0, "InfoNCE should be positive"
    assert not torch.isnan(loss_infonce), "InfoNCE is NaN"
    
    # Shortcut penalty
    sp = ShortcutPenalty(margin=0.25, formulation='confidence_penalty')
    loss_sp = sp(result['blind_scores'], rat_labels)
    assert loss_sp.item() >= 0, "SP should be non-negative"
    assert not torch.isnan(loss_sp), "SP is NaN"
    
    # Combined
    total = loss_infonce + 0.1 * loss_sp
    total.backward()
    
    print(f"  InfoNCE loss: {loss_infonce.item():.4f}")
    print(f"  SP loss: {loss_sp.item():.4f}")
    print(f"  Total: {total.item():.4f}")
    print("✓ Loss integration correct")


if __name__ == "__main__":
    print("\n=== CACR-SP Model Unit Tests ===\n")
    
    print("Test 1: Shape verification")
    test_cacr_sp_model_shapes()
    
    print("\nTest 2: Gradient flow")
    test_cacr_sp_gradient_flow()
    
    print("\nTest 3: Trainable parameters")
    test_cacr_sp_trainable_params()
    
    print("\nTest 4: Loss integration")
    test_loss_integration()
    
    print("\n=== ALL TESTS PASSED ===")
