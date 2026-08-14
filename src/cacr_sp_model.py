import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import BaselineScorer

class CACRSPVCRModel(nn.Module):
    def __init__(self, vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07):
        super().__init__()
        self.vlm = vlm
        
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        self.answer_scorer = BaselineScorer(dropout=scorer_dropout)
        
        self.context_projection = nn.Linear(2304, embedding_dim)
        self.text_residual_projection = nn.Linear(768, embedding_dim)
        self.rationale_projection = nn.Linear(768, embedding_dim)
        self.blind_projection = nn.Linear(768, embedding_dim)
        
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

    def compute_context(self, image_embs, text_embs):
        concat_features = torch.cat(
            [image_embs, text_embs, image_embs * text_embs],
            dim=-1
        )
        interaction_proj = self.context_projection(concat_features)
        text_residual = self.text_residual_projection(text_embs)

        interaction_proj = F.normalize(interaction_proj, p=2, dim=-1)
        text_residual = F.normalize(text_residual, p=2, dim=-1)

        context = interaction_proj + text_residual
        context = F.normalize(context, p=2, dim=-1)
        return context

    def forward_rationale(self, images, questions, selected_answers, rationale_choices, image_embs=None):
        B = len(questions)
        if image_embs is None:
            image_embs = self.encode_images(images)
            
        ctx_texts = [f"Question: {q} Answer: {selected_answers[i]}" for i, q in enumerate(questions)]
        ctx_text_embs = self.vlm.encode_text(ctx_texts)
        
        context_embedding = self.compute_context(image_embs, ctx_text_embs)
        
        rat_texts_flat = []
        for r_list in rationale_choices:
            for r in r_list:
                rat_texts_flat.append(f"Rationale: {r}")
                
        rat_text_embs = self.vlm.encode_text(rat_texts_flat)
        rationale_embeddings_flat = self.rationale_projection(rat_text_embs)
        rationale_embeddings = rationale_embeddings_flat.view(B, 4, -1)
        rationale_embeddings = F.normalize(rationale_embeddings, p=2, dim=-1)
        
        scores = (rationale_embeddings * context_embedding.unsqueeze(1)).sum(-1)
        rationale_scores = scores
        
        blind_embedding = self.blind_projection(ctx_text_embs)
        blind_embedding = F.normalize(blind_embedding, p=2, dim=-1)
        blind_scores_unnorm = (rationale_embeddings.detach() * blind_embedding.unsqueeze(1)).sum(-1)
        blind_scores = blind_scores_unnorm
        
        return {
            "rationale_scores": rationale_scores,
            "blind_scores": blind_scores,
            "context_emb": context_embedding,
            "rationale_embs": rationale_embeddings,
            "context_text_embs": ctx_text_embs
        }

    def forward_blind(self, ctx_text_embs, rationale_embeddings):
        blind_embedding = self.blind_projection(ctx_text_embs)
        blind_embedding = F.normalize(blind_embedding, p=2, dim=-1)
        scores = (rationale_embeddings.detach() * blind_embedding.unsqueeze(1)).sum(-1)
        return scores

    def forward_joint_rationale(self, images, questions, answer_choices, rationale_choices, image_embs=None):
        B = len(questions)
        if image_embs is None:
            image_embs = self.encode_images(images)
            
        # Create context for all 4 answers
        ctx_texts_flat = []
        for q, a_list in zip(questions, answer_choices):
            for a in a_list:
                ctx_texts_flat.append(f"Question: {q} Answer: {a}")
                
        ctx_text_embs = self.vlm.encode_text(ctx_texts_flat) # [B*4, 768]
        
        # Expand image_embs from [B, 768] to [B*4, 768]
        img_embs_expanded = image_embs.unsqueeze(1).expand(B, 4, -1).reshape(B * 4, -1)
        
        context_embedding_flat = self.compute_context(img_embs_expanded, ctx_text_embs)
        context_embedding = context_embedding_flat.view(B, 4, -1) # [B, 4_ans, dim]
        
        rat_texts_flat = []
        for r_list in rationale_choices:
            for r in r_list:
                rat_texts_flat.append(f"Rationale: {r}")
                
        rat_text_embs = self.vlm.encode_text(rat_texts_flat)
        rationale_embeddings_flat = self.rationale_projection(rat_text_embs)
        rationale_embeddings = rationale_embeddings_flat.view(B, 4, -1) # [B, 4_rat, dim]
        rationale_embeddings = F.normalize(rationale_embeddings, p=2, dim=-1)
        
        # We want scores[b, a, r] = context_embedding[b, a] * rationale_embeddings[b, r]
        # context_embedding: [B, 4, 1, dim]
        # rationale_embeddings: [B, 1, 4, dim]
        context_emb_expanded = context_embedding.unsqueeze(2) # [B, 4_ans, 1, dim]
        rat_emb_expanded = rationale_embeddings.unsqueeze(1) # [B, 1, 4_rat, dim]
        
        joint_scores = (context_emb_expanded * rat_emb_expanded).sum(-1) # [B, 4_ans, 4_rat]
        
        # Blind branch
        blind_embedding_flat = self.blind_projection(ctx_text_embs)
        blind_embedding_flat = F.normalize(blind_embedding_flat, p=2, dim=-1)
        blind_embedding = blind_embedding_flat.view(B, 4, -1) # [B, 4_ans, dim]
        
        blind_emb_expanded = blind_embedding.unsqueeze(2) # [B, 4_ans, 1, dim]
        blind_scores = (rat_emb_expanded.detach() * blind_emb_expanded).sum(-1) # [B, 4_ans, 4_rat]
        
        return {
            "joint_rationale_scores": joint_scores,
            "joint_blind_scores": blind_scores,
            "context_embs": context_embedding,
            "rationale_embs": rationale_embeddings,
            "ctx_text_embs": ctx_text_embs
        }
