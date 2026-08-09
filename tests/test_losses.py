"""Test loss functions."""
import pytest
import torch
from losses.contrastive import ContrastiveLoss
from losses.shortcut import ShortcutPenalty
from losses.total_loss import TotalLoss

def test_contrastive_loss_infonce_lower():
    loss_fn = ContrastiveLoss(loss_type='infonce')
    # Correct candidate has highest similarity
    scores = torch.tensor([10.0, 1.0, 1.0, 1.0])
    label = torch.tensor(0)
    loss1 = loss_fn(scores.unsqueeze(0), label.unsqueeze(0))
    
    # Correct candidate has lowest similarity
    scores2 = torch.tensor([1.0, 10.0, 10.0, 10.0])
    loss2 = loss_fn(scores2.unsqueeze(0), label.unsqueeze(0))
    
    assert loss1.item() < loss2.item()

def test_contrastive_loss_margin_ranking():
    loss_fn = ContrastiveLoss(loss_type='margin_ranking', margin=0.5)
    scores = torch.tensor([1.0, 0.5, 0.1, 0.2])
    label = torch.tensor(0)
    loss = loss_fn(scores.unsqueeze(0), label.unsqueeze(0))
    assert loss.item() >= 0

def test_contrastive_loss_gradient():
    loss_fn = ContrastiveLoss(loss_type='infonce')
    scores = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 2])
    loss = loss_fn(scores, labels)
    loss.backward()
    assert scores.grad is not None

def test_shortcut_penalty_confidence():
    penalty_fn = ShortcutPenalty(formulation='confidence_penalty')
    # Uniform blind scores -> low penalty
    blind_scores_uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    label = torch.tensor([0])
    penalty_low = penalty_fn(blind_scores_uniform, label)
    
    # Confident blind scores -> high penalty
    blind_scores_confident = torch.tensor([[0.9, 0.03, 0.03, 0.04]])
    penalty_high = penalty_fn(blind_scores_confident, label)
    
    assert penalty_low.item() < penalty_high.item()

def test_shortcut_penalty_gradient():
    penalty_fn = ShortcutPenalty(formulation='confidence_penalty')
    blind_scores = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    penalty = penalty_fn(blind_scores, labels)
    penalty.backward()
    assert blind_scores.grad is not None

def test_total_loss_combines():
    c_loss = ContrastiveLoss(loss_type='infonce')
    s_pen = ShortcutPenalty(formulation='confidence_penalty')
    total_fn = TotalLoss(contrastive_loss=c_loss, shortcut_penalty=s_pen, lambda_sp=0.1)
    
    r_scores = torch.randn(2, 4)
    b_scores = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    
    out = total_fn(r_scores, labels, b_scores)
    assert 'total' in out
    assert 'contrastive' in out
    assert 'shortcut' in out
    assert out['total'] == out['contrastive'] + 0.1 * out['shortcut']

def test_total_loss_without_shortcut():
    c_loss = ContrastiveLoss(loss_type='infonce')
    total_fn = TotalLoss(contrastive_loss=c_loss, shortcut_penalty=None, lambda_sp=0.1)
    
    r_scores = torch.randn(2, 4)
    labels = torch.tensor([0, 1])
    
    out = total_fn(r_scores, labels, None)
    assert out['shortcut'].item() == 0.0
    assert out['total'] == out['contrastive']

def test_total_loss_gradient():
    c_loss = ContrastiveLoss(loss_type='infonce')
    s_pen = ShortcutPenalty(formulation='confidence_penalty')
    total_fn = TotalLoss(contrastive_loss=c_loss, shortcut_penalty=s_pen, lambda_sp=0.1)
    
    r_scores = torch.randn(2, 4, requires_grad=True)
    b_scores = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    
    out = total_fn(r_scores, labels, b_scores)
    out['total'].backward()
    
    assert r_scores.grad is not None
    assert b_scores.grad is not None
