import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import BaselineScorer

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        # L2 norm
        return x / x.norm(p=2, dim=-1, keepdim=True)

class CACRSPVCRModel(nn.Module):
    def __init__(self, vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07):
        super().__init__()
        self.vlm = vlm
        
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        self.answer_scorer = BaselineScorer(dropout=scorer_dropout)
        
        self.img_proj = ProjectionHead(768, 1024, embedding_dim)
        self.txt_proj = ProjectionHead(768, 1024, embedding_dim)
        self.rationale_projection = ProjectionHead(768, 1024, embedding_dim)
        
        self.temperature = temperature

    def encode_images(self, images):
        return self.vlm.encode_image(images)

    def forward_answer(self, images, questions, answer_choices, image_embs=None):
        B = len(questions)
        if image_embs is None:
            image_embs = self.encode_images(images)
            
        texts_flat = []
        for q, a_list in zip(questions, answer_choices):
            for a in a_list:
                texts_flat.append(f"Question: {q} Answer: {a}")
                
        text_embs = self.vlm.encode_text(texts_flat)
        img_embs_expanded = image_embs.unsqueeze(1).expand(B, 4, -1).reshape(B * 4, -1)
        
        scores = self.answer_scorer(img_embs_expanded, text_embs)
        return scores.view(B, 4)

    def forward_rationale(self, images, questions, selected_answers, rationale_choices, image_embs=None):
        B = len(questions)
        if image_embs is None:
            image_embs = self.encode_images(images)
            
        ctx_texts = [f"Question: {q} Answer: {selected_answers[i]}" for i, q in enumerate(questions)]
        ctx_text_embs = self.vlm.encode_text(ctx_texts)
        
        context_embedding = F.normalize(self.img_proj(image_embs) + self.txt_proj(ctx_text_embs), p=2, dim=-1)
        
        rat_texts_flat = []
        for r_list in rationale_choices:
            for r in r_list:
                rat_texts_flat.append(f"Rationale: {r}")
                
        rat_text_embs = self.vlm.encode_text(rat_texts_flat)
        rationale_embeddings_flat = self.rationale_projection(rat_text_embs)
        rationale_embeddings = rationale_embeddings_flat.view(B, 4, -1)
        
        scores = (rationale_embeddings * context_embedding.unsqueeze(1)).sum(-1)
        rationale_scores = scores
        
        # Compute blind scores by routing zeroed images through the main context projection
        zero_image_embs = torch.zeros_like(image_embs)
        blind_context_embedding = F.normalize(self.img_proj(zero_image_embs) + self.txt_proj(ctx_text_embs), p=2, dim=-1)
        
        # Do not detach rationale_embeddings so SP gradients flow into the rationale projection
        blind_scores = (rationale_embeddings * blind_context_embedding.unsqueeze(1)).sum(-1)
        
        return {
            "rationale_scores": rationale_scores,
            "blind_scores": blind_scores,
            "context_emb": context_embedding,
            "rationale_embs": rationale_embeddings,
            "context_text_embs": ctx_text_embs
        }

    def forward_blind(self, ctx_text_embs, rationale_embeddings):
        B = ctx_text_embs.size(0)
        # Dummy zero image embs for blind evaluation
        zero_image_embs = torch.zeros(B, 768, device=ctx_text_embs.device, dtype=ctx_text_embs.dtype)
        blind_context_embedding = F.normalize(self.img_proj(zero_image_embs) + self.txt_proj(ctx_text_embs), p=2, dim=-1)
        
        scores = (rationale_embeddings * blind_context_embedding.unsqueeze(1)).sum(-1)
        return scores
