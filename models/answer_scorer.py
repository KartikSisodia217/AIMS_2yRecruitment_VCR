"""
Answer Scorer models for Stage 1.
"""
from abc import ABC, abstractmethod
import torch
from typing import Any, List
from .vlm_backbone import VLMBackbone


class AnswerScorer(ABC):
    """Abstract interface for Stage-1 answer scoring.
    
    Given an image and question, scores each of the 4 candidate answers.
    The exact scoring mechanism is a RESEARCH DECISION.
    # RESEARCH_DECISION: scoring mechanism
    """
    
    @abstractmethod
    def score_candidates(self, image: Any, question: str, 
                         candidates: List[str]) -> torch.Tensor:
        """Score each candidate answer.
        
        Args:
            image: PIL Image or None
            question: Question text
            candidates: List of 4 candidate answer texts
        
        Returns:
            Tensor of shape [4] with scores (higher = better)
        """
        ...
    
    def predict(self, image: Any, question: str, 
                candidates: List[str]) -> int:
        """Return index of highest-scoring candidate."""
        scores = self.score_candidates(image, question, candidates)
        return scores.argmax().item()


class LogLikelihoodScorer(AnswerScorer):
    """Score candidates by generative log-likelihood.
    
    For each candidate, computes P(candidate | image, question) using
    the VLM's autoregressive likelihood.
    """
    def __init__(self, vlm: VLMBackbone):
        self.vlm = vlm
    
    def score_candidates(self, image: Any, question: str, candidates: List[str]) -> torch.Tensor:
        scores = []
        for candidate in candidates:
            ll = self.vlm.compute_log_likelihood(image, question, candidate)
            scores.append(ll)
        return torch.tensor(scores, dtype=torch.float32)


class MockAnswerScorer(AnswerScorer):
    """Deterministic mock scorer for testing.
    Returns scores where the candidate matching a specified label scores highest.
    """
    def __init__(self, default_prediction: int = 0):
        self.default_prediction = default_prediction
    
    def score_candidates(self, image: Any, question: str, candidates: List[str]) -> torch.Tensor:
        scores = torch.tensor([0.1, 0.1, 0.1, 0.1])
        scores[self.default_prediction] = 0.9
        return scores
