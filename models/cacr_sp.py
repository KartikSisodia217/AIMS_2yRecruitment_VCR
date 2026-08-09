"""
CACR-SP Top-level model assembly.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Any
from .vlm_backbone import VLMBackbone
from .answer_scorer import AnswerScorer
from .projection import ProjectionHead
from .rationale_encoder import RationaleEncoder
from .similarity import SimilarityFunction


class CACRSPModel(nn.Module):
    """Context-Anchored Contrastive Ranking with Shortcut Penalty.
    
    This is the full CACR-SP model assembly. It orchestrates:
    1. VLM backbone for multimodal encoding
    2. Answer scorer for Stage 1 (answer selection)
    3. Projection head for context embedding
    4. Rationale encoder for rationale embeddings
    5. Similarity function for ranking
    
    NOTE: This is a research architecture under development.
    Not all components are finalized.
    # RESEARCH_DECISION: many design choices remain open
    """
    
    def __init__(self, vlm: VLMBackbone, answer_scorer: AnswerScorer,
                 projection: ProjectionHead, rationale_encoder: RationaleEncoder,
                 similarity_fn: SimilarityFunction, config: dict = None):
        super().__init__()
        self.vlm = vlm
        self.answer_scorer = answer_scorer
        self.projection = projection
        self.rationale_encoder = rationale_encoder
        self.similarity_fn = similarity_fn
        self.config = config or {}
    
    def predict_answer(self, image: Any, question: str, answer_choices: List[str]) -> Tuple[int, torch.Tensor]:
        """Stage 1: Select answer from candidates.
        
        Returns:
            (predicted_answer_index, answer_scores)
        """
        scores = self.answer_scorer.score_candidates(image, question, answer_choices)
        pred = scores.argmax().item()
        return pred, scores
    
    def score_rationales(self, image: Any, question: str, answer: str, 
                         rationale_choices: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 2: Score rationale candidates using CACR.
        
        Args:
            image: PIL Image
            question: Question text
            answer: Selected answer text
            rationale_choices: List of 4 rationale texts
        
        Returns:
            (similarity_scores [4], context_embedding [emb_dim], rationale_embeddings [4, emb_dim])
        """
        # 1. Get VLM hidden representation for Image + Question + Answer
        context_text = f"{question} {answer}"  # RESEARCH_DECISION: formatting
        hidden = self.vlm.encode(image, context_text)
        
        # 2. Project to embedding space
        context_emb = self.projection(hidden.unsqueeze(0)).squeeze(0)
        
        # 3. Encode rationales
        rationale_embs = self.rationale_encoder.encode(rationale_choices)
        
        # 4. Compute similarities
        scores = self.similarity_fn.compute(context_emb, rationale_embs)
        
        return scores, context_emb, rationale_embs
    
    def predict_rationale(self, image: Any, question: str, answer: str,
                          rationale_choices: List[str]) -> Tuple[int, torch.Tensor]:
        """Predict rationale (convenience method)."""
        scores, _, _ = self.score_rationales(image, question, answer, rationale_choices)
        pred = scores.argmax().item()
        return pred, scores
    
    def forward_blind(self, question: str, candidate: str) -> torch.Tensor:
        """Blind branch: encode without image (for shortcut penalty).
        
        Training only. Returns hidden representation without visual info.
        """
        return self.vlm.encode_text_only(f"{question} {candidate}")
    
    def predict(self, image: Any, question: str, answer_choices: List[str], 
                rationale_choices: List[str], use_gt_answer: bool = False,
                gt_answer_idx: Optional[int] = None) -> dict:
        """Full pipeline: predict both answer and rationale.
        
        Args:
            image: PIL Image
            question: Question text
            answer_choices: List of 4 answer texts
            rationale_choices: List of 4 rationale texts
            use_gt_answer: If True, use gt_answer_idx for Stage 2 (debug/training mode)
            gt_answer_idx: Ground-truth answer index (required if use_gt_answer=True)
        
        Returns:
            dict with: predicted_answer, predicted_rationale, answer_scores, rationale_scores
        """
        # Stage 1
        answer_pred, answer_scores = self.predict_answer(image, question, answer_choices)
        
        # Select answer for Stage 2
        if use_gt_answer and gt_answer_idx is not None:
            selected_answer_idx = gt_answer_idx
        else:
            selected_answer_idx = answer_pred
        selected_answer = answer_choices[selected_answer_idx]
        
        # Stage 2
        rationale_pred, rationale_scores = self.predict_rationale(
            image, question, selected_answer, rationale_choices
        )
        
        return {
            'predicted_answer': answer_pred,
            'predicted_rationale': rationale_pred,
            'answer_scores': answer_scores,
            'rationale_scores': rationale_scores,
            'selected_answer_idx': selected_answer_idx,
            'selected_answer_text': selected_answer,
        }
