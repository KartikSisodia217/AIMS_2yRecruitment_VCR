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
        
        self.context_projection = ProjectionHead(2304, 1024, embedding_dim)
        self.rationale_projection = ProjectionHead(768, 1024, embedding_dim)
        self.blind_projection = ProjectionHead(768, 1024, embedding_dim)
        
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
            
        ctx_texts = [f"{q} {selected_answers[i]}" for i, q in enumerate(questions)]
        ctx_text_embs = self.vlm.encode_text(ctx_texts)
        
        h_context = torch.cat([image_embs, ctx_text_embs, image_embs * ctx_text_embs], dim=-1)
        context_embedding = self.context_projection(h_context)
        
        rat_texts_flat = []
        for r_list in rationale_choices:
            for r in r_list:
                rat_texts_flat.append(r)
                
        rat_text_embs = self.vlm.encode_text(rat_texts_flat)
        rationale_embeddings_flat = self.rationale_projection(rat_text_embs)
        rationale_embeddings = rationale_embeddings_flat.view(B, 4, -1)
        
        scores = (rationale_embeddings * context_embedding.unsqueeze(1)).sum(-1)
        rationale_scores = scores / self.temperature
        
        blind_embedding = self.blind_projection(ctx_text_embs)
        blind_scores_unnorm = (rationale_embeddings * blind_embedding.unsqueeze(1)).sum(-1)
        blind_scores = blind_scores_unnorm / self.temperature
        
        return {
            "rationale_scores": rationale_scores,
            "blind_scores": blind_scores,
            "context_emb": context_embedding,
            "rationale_embs": rationale_embeddings,
            "context_text_embs": ctx_text_embs
        }

    def forward_blind(self, ctx_text_embs, rationale_embeddings):
        blind_embedding = self.blind_projection(ctx_text_embs)
        scores = (rationale_embeddings * blind_embedding.unsqueeze(1)).sum(-1)
        return scores / self.temperature
