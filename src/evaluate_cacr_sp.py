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
            
        result_joint = model.forward_joint_rationale(
            images, questions, answer_choices, rationale_choices, image_embs=image_embs
        )
        
        # Extract TF scores
        B = len(ans_labels)
        joint_rat_scores = result_joint["joint_rationale_scores"]
        rat_scores_tf = joint_rat_scores[torch.arange(B), ans_labels, :]
        rat_preds_tf = rat_scores_tf.argmax(dim=-1)
        correct_r_tf += (rat_preds_tf == rat_labels).sum().item()
        
        # 4. QA->R (Predicted Answer) - using the joint tensor
        rat_scores_pred = joint_rat_scores[torch.arange(B), ans_preds, :]
        rat_preds_pred = rat_scores_pred.argmax(dim=-1)
        match_r_pred = (rat_preds_pred == rat_labels)
        correct_r_pred += match_r_pred.sum().item()
        
        # 5. Q->AR (Joint Inference)
        # joint_scores[b, a, r] = ans_logits[b, a] + (joint_rat_scores[b, a, r] / 0.07)
        # assuming temperature = model.temperature
        temperature = getattr(model, 'temperature', 0.07)
        scaled_joint_rat_scores = joint_rat_scores / temperature
        joint_scores = ans_logits.unsqueeze(2) + scaled_joint_rat_scores # [B, 4, 4]
        
        joint_scores_flat = joint_scores.view(B, 16)
        joint_preds_flat = joint_scores_flat.argmax(dim=-1)
        
        joint_ans_preds = joint_preds_flat // 4
        joint_rat_preds = joint_preds_flat % 4
        
        match_joint_ar = (joint_ans_preds == ans_labels) & (joint_rat_preds == rat_labels)
        correct_ar += match_joint_ar.sum().item()
        
        # 6. Blind Branch (using teacher forced context)
        joint_blind_scores = result_joint["joint_blind_scores"]
        blind_scores = joint_blind_scores[torch.arange(B), ans_labels, :]
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
