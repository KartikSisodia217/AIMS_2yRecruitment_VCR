import torch
from src.utils import load_image

@torch.no_grad()
def evaluate_model(model, dataloader, device, zip_path="data/vcr/vcr1images.zip"):
    """
    Evaluates the model on a validation/test split.
    Metrics computed:
    - Q->A accuracy
    - QA->R accuracy (using predicted answer A*)
    - Q->AR joint accuracy
    """
    model.eval()
    
    total = 0
    correct_a = 0
    correct_r = 0
    correct_ar = 0
    
    for batch in dataloader:
        images = [load_image(p, zip_path) for p in batch["image_path"]]
        questions = batch["question"]
        answer_choices = batch["answer_choices"]
        ans_labels = batch["answer_label"].to(device)
        
        # Encode images ONCE and reuse for both answer + rationale
        image_embs = model.encode_images(images)
        
        # 1. Predict Answer
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_preds = ans_logits.argmax(dim=-1)
        
        # 2. Extract predicted answer text for rationale scoring
        # CRITICAL: We use the predicted answer A*, not the ground truth
        selected_answers = []
        for i in range(len(ans_preds)):
            pred_idx = ans_preds[i].item()
            selected_answers.append(answer_choices[i][pred_idx])
            
        # 3. Predict Rationale
        rationale_choices = batch["rationale_choices"]
        rat_labels = batch["rationale_label"].to(device)
        rat_logits = model.forward_rationale(images, questions, selected_answers, rationale_choices, image_embs=image_embs)
        rat_preds = rat_logits.argmax(dim=-1)
        
        # 4. Update metrics
        total += len(ans_preds)
        
        match_a = (ans_preds == ans_labels)
        match_r = (rat_preds == rat_labels)
        match_ar = match_a & match_r
        
        correct_a += match_a.sum().item()
        correct_r += match_r.sum().item()
        correct_ar += match_ar.sum().item()
        
    metrics = {
        "acc_a": correct_a / total if total > 0 else 0.0,
        "acc_r": correct_r / total if total > 0 else 0.0,
        "acc_ar": correct_ar / total if total > 0 else 0.0,
    }
    
    return metrics
