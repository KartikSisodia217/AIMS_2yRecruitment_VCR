import torch
import torch.nn as nn

class BaselineScorer(nn.Module):
    def __init__(self, input_dim=768, hidden1=512, hidden2=128, dropout=0.1):
        """
        A simple MLP baseline scorer that fuses image and text embeddings.
        Fusion = [image_emb, text_emb, image_emb * text_emb]
        """
        super().__init__()
        fusion_dim = input_dim * 3
        
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
        
    def forward(self, image_emb, text_emb):
        """
        :param image_emb: [batch_size, input_dim]
        :param text_emb: [batch_size, input_dim]
        :return: [batch_size, 1] scalar score for each pair
        """
        fused = torch.cat([image_emb, text_emb, image_emb * text_emb], dim=-1)
        score = self.mlp(fused)
        return score


class BaselineVCRModel(nn.Module):
    def __init__(self, vlm, scorer_dropout=0.1):
        """
        Initializes the overall Baseline VCR model.
        :param vlm: The SigLIP2Wrapper instance (will be frozen)
        """
        super().__init__()
        self.vlm = vlm
        
        # Freeze VLM
        for param in self.vlm.parameters():
            param.requires_grad = False
            
        # Separate scorers for answers and rationales (distinct 4-way classification tasks)
        self.answer_scorer = BaselineScorer(dropout=scorer_dropout)
        self.rationale_scorer = BaselineScorer(dropout=scorer_dropout)
    
    def encode_images(self, images):
        """
        Encode a batch of images through the frozen VLM vision encoder.
        This should be called ONCE per batch and the result reused.
        
        :param images: list of PIL images (length B)
        :return: tensor of shape [B, embed_dim] (L2-normalized)
        """
        return self.vlm.encode_image(images)
    
    def _score_candidates(self, image_embs, texts_flat, scorer, B, num_candidates=4):
        """
        Internal helper: score candidates given precomputed image embeddings.
        
        :param image_embs: [B, embed_dim] — one embedding per sample
        :param texts_flat: list of B*num_candidates text strings
        :param scorer: the BaselineScorer to use
        :param B: batch size
        :param num_candidates: number of candidates per sample (default 4)
        :return: logits of shape [B, num_candidates]
        """
        # Encode all candidate texts at once
        text_embs = self.vlm.encode_text(texts_flat)  # [B*num_candidates, embed_dim]
        
        # Expand image embeddings to match candidates:
        # [B, embed_dim] → [B, 1, embed_dim] → [B, num_candidates, embed_dim] → [B*num_candidates, embed_dim]
        img_embs_expanded = image_embs.unsqueeze(1).expand(B, num_candidates, -1).reshape(B * num_candidates, -1)
        
        # Score
        scores = scorer(img_embs_expanded, text_embs)  # [B*num_candidates, 1]
        logits = scores.view(B, num_candidates)
        return logits
        
    def forward_answer(self, images, questions, answer_choices, image_embs=None):
        """
        Forward pass for answer selection.
        
        :param images: list of PIL images (length B) — ignored if image_embs provided
        :param questions: list of strings (length B)
        :param answer_choices: list of lists of 4 strings (B x 4)
        :param image_embs: optional precomputed image embeddings [B, embed_dim]
        :return: logits of shape [B, 4]
        """
        B = len(questions)
        
        # Encode images (reuse if precomputed)
        if image_embs is None:
            image_embs = self.encode_images(images)
        
        # Build candidate texts
        texts_flat = []
        for q, a_list in zip(questions, answer_choices):
            for a in a_list:
                texts_flat.append(f"Question: {q} Answer: {a}")
                
        return self._score_candidates(image_embs, texts_flat, self.answer_scorer, B)

    def forward_rationale(self, images, questions, selected_answers, rationale_choices, image_embs=None):
        """
        Forward pass for rationale selection.
        NOTE ON TRAINING VS INFERENCE (Information Leakage):
        The caller must decide whether `selected_answers` is the Ground Truth answer (teacher forcing) 
        or the predicted answer A* (inference). 
        To avoid silent assumptions, this method explicitly requires the answers to be provided 
        by the caller rather than internally choosing ground truth vs predictions.
        
        :param images: list of PIL images (length B) — ignored if image_embs provided
        :param questions: list of strings (length B)
        :param selected_answers: list of strings representing the chosen answers (length B)
        :param rationale_choices: list of lists of 4 strings (B x 4)
        :param image_embs: optional precomputed image embeddings [B, embed_dim]
        :return: logits of shape [B, 4]
        """
        B = len(questions)
        
        # Encode images (reuse if precomputed)
        if image_embs is None:
            image_embs = self.encode_images(images)
        
        # Build candidate texts
        texts_flat = []
        for q, ans, r_list in zip(questions, selected_answers, rationale_choices):
            for r in r_list:
                texts_flat.append(f"Question: {q} Answer: {ans} Rationale: {r}")
                
        return self._score_candidates(image_embs, texts_flat, self.rationale_scorer, B)

