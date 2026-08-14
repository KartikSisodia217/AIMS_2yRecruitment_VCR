import torch
from src.utils import load_image

@torch.no_grad()
def evaluate_cacr_sp(model, dataloader, device, zip_path="data/vcr/vcr1images.zip"):
    """
    Evaluates the CACR-SP model.
    Metrics computed:
    - Q->A accuracy
    - QA->R (Teacher-forced) accuracy
    - QA->R (Predicted answer) accuracy
    - Q->AR joint accuracy (uses predicted answer)
    - Blind branch accuracy
    """
    model.eval()
    
    total = 0
    correct_a = 0
    correct_r_tf = 0
    correct_r_pred = 0
    correct_ar = 0
    correct_blind = 0
    
    for batch in dataloader:
        images = [load_image(p, zip_path) for p in batch["image_path"]]
        questions = batch["question"]
        answer_choices = batch["answer_choices"]
        ans_labels = batch["answer_label"].to(device)
        rationale_choices = batch["rationale_choices"]
        rat_labels = batch["rationale_label"].to(device)
        
        # 1. Encode images ONCE
        image_embs = model.encode_images(images)
        
        # 2. Predict Answer
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_preds = ans_logits.argmax(dim=-1)
        match_a = (ans_preds == ans_labels)
        correct_a += match_a.sum().item()
        
        # 3. QA->R (Teacher-Forced)
        gt_answers = []
        for i, label_idx in enumerate(ans_labels):
            gt_answers.append(answer_choices[i][label_idx.item()])
            
        result_tf = model.forward_rationale(
            images, questions, gt_answers, rationale_choices, image_embs=image_embs
        )
        rat_scores_tf = result_tf["rationale_scores"]
        rat_embs = result_tf["rationale_embs"]
        ctx_text_embs_tf = result_tf["context_text_embs"]
        rat_preds_tf = rat_scores_tf.argmax(dim=-1)
        correct_r_tf += (rat_preds_tf == rat_labels).sum().item()
        
        # 4. QA->R (Predicted Answer)
        pred_answers = []
        for i, pred_idx in enumerate(ans_preds):
            pred_answers.append(answer_choices[i][pred_idx.item()])
            
        result_pred = model.forward_rationale(
            images, questions, pred_answers, rationale_choices, image_embs=image_embs
        )
        rat_scores_pred = result_pred["rationale_scores"]
        rat_preds_pred = rat_scores_pred.argmax(dim=-1)
        match_r_pred = (rat_preds_pred == rat_labels)
        correct_r_pred += match_r_pred.sum().item()
        
        # 5. Q->AR
        match_ar = match_a & match_r_pred
        correct_ar += match_ar.sum().item()
        
        # 6. Blind Branch (using teacher forced context)
        blind_scores = result_tf["blind_scores"]
        blind_preds = blind_scores.argmax(dim=-1)
        correct_blind += (blind_preds == rat_labels).sum().item()
        
        total += len(ans_preds)
        
    metrics = {
        "acc_a": correct_a / total if total > 0 else 0.0,
        "acc_r_tf": correct_r_tf / total if total > 0 else 0.0,
        "acc_r_pred": correct_r_pred / total if total > 0 else 0.0,
        "acc_ar": correct_ar / total if total > 0 else 0.0,
        "acc_blind": correct_blind / total if total > 0 else 0.0,
    }
    
    return metrics
