import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.model import BaselineVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image
from src.evaluate import evaluate_model

def train_one_epoch(model, dataloader, optimizer, criterion, device, args):
    model.train()
    
    total_loss = 0.0
    total_ans_loss = 0.0
    total_rat_loss = 0.0
    
    correct_a = 0
    correct_r = 0
    correct_ar = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        images = [load_image(p, args.zip_path) for p in batch["image_path"]]
        questions = batch["question"]
        
        answer_choices = batch["answer_choices"]
        ans_labels = batch["answer_label"].to(device)
        
        rationale_choices = batch["rationale_choices"]
        rat_labels = batch["rationale_label"].to(device)
        
        optimizer.zero_grad()
        
        # --- Encode images ONCE and reuse for both answer + rationale ---
        image_embs = model.encode_images(images)
        
        # --- Answer Forward ---
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_loss = criterion(ans_logits, ans_labels)
        
        # --- Rationale Forward (Teacher Forcing) ---
        # For training, we use ground truth answers
        gt_answers = []
        for i, label_idx in enumerate(ans_labels):
            gt_answers.append(answer_choices[i][label_idx.item()])
            
        rat_logits = model.forward_rationale(images, questions, gt_answers, rationale_choices, image_embs=image_embs)
        rat_loss = criterion(rat_logits, rat_labels)
        
        # --- Loss ---
        loss = (args.ans_loss_weight * ans_loss) + (args.rat_loss_weight * rat_loss)
        
        # --- Backward ---
        loss.backward()
        optimizer.step()
        
        # --- Metrics tracking (Training) ---
        total_loss += loss.item()
        total_ans_loss += ans_loss.item()
        total_rat_loss += rat_loss.item()
        
        ans_preds = ans_logits.argmax(dim=-1)
        rat_preds = rat_logits.argmax(dim=-1)
        
        match_a = (ans_preds == ans_labels)
        match_r = (rat_preds == rat_labels)
        
        correct_a += match_a.sum().item()
        correct_r += match_r.sum().item()
        correct_ar += (match_a & match_r).sum().item()
        total_samples += len(ans_labels)
        
        if batch_idx % args.log_interval == 0:
            print(f"Batch [{batch_idx}/{len(dataloader)}] - Loss: {loss.item():.4f}")
            
    epoch_time = time.time() - start_time
    metrics = {
        "loss": total_loss / len(dataloader),
        "ans_loss": total_ans_loss / len(dataloader),
        "rat_loss": total_rat_loss / len(dataloader),
        "acc_a": correct_a / total_samples,
        "acc_r": correct_r / total_samples,
        "acc_ar": correct_ar / total_samples,
        "time": epoch_time
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Baseline VCR Model")
    parser.add_argument("--data_dir", type=str, default="data/vcr", help="VCR data directory")
    parser.add_argument("--zip_path", type=str, default="data/vcr/vcr1images.zip", help="Zip path for images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Scorer dropout")
    parser.add_argument("--ans_loss_weight", type=float, default=1.0, help="Weight for answer loss")
    parser.add_argument("--rat_loss_weight", type=float, default=1.0, help="Weight for rationale loss")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="Subset for tiny overfit test")
    parser.add_argument("--max_val_samples", type=int, default=-1, help="Subset for validation")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # --- Datasets ---
    print("Loading datasets...")
    train_dataset = VCRDataset(split="train", data_dir=args.data_dir)
    val_dataset = VCRDataset(split="val", data_dir=args.data_dir)
    
    if args.max_train_samples > 0:
        indices = list(range(min(args.max_train_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        print(f"Subsampled train set to {len(train_dataset)} samples.")
        
    if args.max_val_samples > 0:
        indices = list(range(min(args.max_val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
        print(f"Subsampled val set to {len(val_dataset)} samples.")
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=vcr_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=vcr_collate_fn
    )
    
    # --- Models ---
    print("Initializing models...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device.type)
    model = BaselineVCRModel(vlm=vlm, scorer_dropout=args.dropout)
    model.to(device)
    
    # --- Optimizer & Loss ---
    # Only optimize scoring MLP parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    best_val_acc = -1.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, args)
        
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Q->A Acc:  {train_metrics['acc_a']:.4f}")
        print(f"Train QA->R Acc: {train_metrics['acc_r']:.4f} (Teacher-forced)")
        print(f"Train Q->AR Acc: {train_metrics['acc_ar']:.4f} (Teacher-forced)")
        print(f"Time: {train_metrics['time']:.2f}s")
        
        print("\nRunning validation...")
        val_metrics = evaluate_model(model, val_loader, device, zip_path=args.zip_path)
        
        print(f"Val Q->A Acc:  {val_metrics['acc_a']:.4f}")
        print(f"Val QA->R Acc: {val_metrics['acc_r']:.4f} (Predicted answer)")
        print(f"Val Q->AR Acc: {val_metrics['acc_ar']:.4f} (Joint Inference)")
        
        # Save best
        if val_metrics["acc_ar"] > best_val_acc:
            best_val_acc = val_metrics["acc_ar"]
            save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            
            # Only save the trainable scorer parameters, not the huge frozen VLM
            state_dict = {
                "answer_scorer": model.answer_scorer.state_dict(),
                "rationale_scorer": model.rationale_scorer.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics
            }
            torch.save(state_dict, save_path)
            print(f"Saved best model with Val Q->AR: {best_val_acc:.4f} to {save_path}")

if __name__ == "__main__":
    main()
