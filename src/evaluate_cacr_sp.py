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

if __name__ == "__main__":
    import argparse
    import os
    from torch.utils.data import DataLoader, Subset
    from src.dataset import VCRDataset
    from src.vlm import SigLIP2Wrapper
    from src.cacr_sp_model import CACRSPVCRModel
    from src.utils import vcr_collate_fn

    parser = argparse.ArgumentParser(description="Evaluate CACR-SP Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/vcr", help="VCR data directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Root directory containing vcr1images / movie folders.")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Subset for validation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    print("Initializing model...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device.type)
    model = CACRSPVCRModel(vlm=vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07)
    model.to(device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    print("Loading validation dataset...")
    val_dataset = VCRDataset(split="val", data_dir=args.data_dir, image_dir=args.image_dir)
    
    if args.max_val_samples is not None and args.max_val_samples > 0:
        indices = list(range(min(args.max_val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
        print(f"Subsampled val set to {len(val_dataset)} samples.")
        
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vcr_collate_fn,
        num_workers=0
    )

    zip_path = os.path.join(args.data_dir, "vcr1images.zip")

    print("Evaluating...")
    metrics = evaluate_cacr_sp(model, val_dataloader, device, zip_path=zip_path)

    print(f"Val Q->A Acc: {metrics['acc_a']:.4f}")
    print(f"Val QA->R (TF): {metrics['acc_r_tf']:.4f}")
    print(f"Val QA->R (Pred): {metrics['acc_r_pred']:.4f}")
    print(f"Val Q->AR Acc: {metrics['acc_ar']:.4f}")
    print(f"Val Blind Acc: {metrics['acc_blind']:.4f}")
