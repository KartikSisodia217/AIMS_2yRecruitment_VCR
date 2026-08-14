import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image
from losses.contrastive import ContrastiveLoss
from losses.shortcut import ShortcutPenalty

# We'll use the evaluate function for CACR-SP which we will create shortly.
from src.evaluate_cacr_sp import evaluate_cacr_sp

def train_one_epoch(model, dataloader, optimizer, criterion, contrastive_criterion, sp_criterion, device, args, epoch, best_val_acc, total_batches, start_batch_idx=0, metrics_state=None):
    model.train()
    
    if metrics_state:
        total_loss = metrics_state.get("total_loss", 0.0)
        total_ans_loss = metrics_state.get("total_ans_loss", 0.0)
        total_rat_loss = metrics_state.get("total_rat_loss", 0.0)
        total_sp_loss = metrics_state.get("total_sp_loss", 0.0)
        correct_a = metrics_state.get("correct_a", 0)
        correct_r = metrics_state.get("correct_r", 0)
        correct_ar = metrics_state.get("correct_ar", 0)
        total_samples = metrics_state.get("total_samples", 0)
    else:
        total_loss = 0.0
        total_ans_loss = 0.0
        total_rat_loss = 0.0
        total_sp_loss = 0.0
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
        
        image_embs = model.encode_images(images)
        
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_loss = criterion(ans_logits, ans_labels)
        
        gt_answers = []
        for i, label_idx in enumerate(ans_labels):
            gt_answers.append(answer_choices[i][label_idx.item()])
            
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            result = model.forward_joint_rationale(
                images, questions, answer_choices, rationale_choices, image_embs=image_embs
            )
            
        # Extract teacher-forced rationale scores for SP and logging
        # We need the scores where answer == ans_label
        joint_rat_scores = result["joint_rationale_scores"] # [B, 4, 4]
        B = len(ans_labels)
        rat_scores_tf = joint_rat_scores[torch.arange(B), ans_labels, :] # [B, 4]
        
        rat_loss = contrastive_criterion(rat_scores_tf, rat_labels)
        
        # 16-way Joint Loss
        # joint_scores[b, a, r] = ans_logits[b, a] + (joint_rat_scores[b, a, r] / args.temperature)
        scaled_joint_rat_scores = joint_rat_scores / args.temperature
        joint_scores = ans_logits.unsqueeze(2) + scaled_joint_rat_scores # [B, 4, 4]
        joint_scores_flat = joint_scores.view(B, 16)
        joint_labels = ans_labels * 4 + rat_labels
        joint_loss = F.cross_entropy(joint_scores_flat, joint_labels)
        
        sp_loss = torch.tensor(0.0).to(device)
        if args.enable_sp:
            joint_blind_scores = result["joint_blind_scores"]
            blind_scores_tf = joint_blind_scores[torch.arange(B), ans_labels, :]
            sp_loss = sp_criterion(blind_scores_tf, rat_labels)
            
            
        loss = (args.ans_loss_weight * ans_loss) + joint_loss + (args.lambda_sp * sp_loss)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ans_loss += ans_loss.item()
        total_rat_loss += rat_loss.item()
        total_sp_loss += sp_loss.item()
        
        ans_preds = ans_logits.argmax(dim=-1)
        rat_preds = rat_scores_tf.argmax(dim=-1)
        
        match_a = (ans_preds == ans_labels)
        match_r = (rat_preds == rat_labels)
        
        correct_a += match_a.sum().item()
        correct_r += match_r.sum().item()
        correct_ar += (match_a & match_r).sum().item()
        total_samples += len(ans_labels)
        
        actual_batch_idx = start_batch_idx + batch_idx
        
        if actual_batch_idx % args.log_interval == 0:
            print(f"Batch [{actual_batch_idx}/{total_batches}] | Total: {loss.item():.4f} | Answer: {ans_loss.item():.4f} | Contrastive: {rat_loss.item():.4f} | SP: {sp_loss.item():.4f}")
            
        if actual_batch_idx > 0 and (actual_batch_idx + 1) % args.checkpoint_every == 0:
            temp_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt.tmp")
            final_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
            
            global_step = (epoch - 1) * total_batches + actual_batch_idx + 1
            print(f"Checkpoint saved:\nepoch={epoch}\nbatch={actual_batch_idx + 1}/{total_batches}\nglobal_step={global_step}\npath={final_path}")
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch_idx": actual_batch_idx,
                "best_val_acc": best_val_acc,
                "metrics_state": {
                    "total_loss": total_loss,
                    "total_ans_loss": total_ans_loss,
                    "total_rat_loss": total_rat_loss,
                    "total_sp_loss": total_sp_loss,
                    "correct_a": correct_a,
                    "correct_r": correct_r,
                    "correct_ar": correct_ar,
                    "total_samples": total_samples
                },
                "torch_rng_state": torch.get_rng_state()
            }, temp_path)
            os.replace(temp_path, final_path)
            
    epoch_time = time.time() - start_time
    metrics = {
        "loss": total_loss / total_batches if total_batches > 0 else 0,
        "ans_loss": total_ans_loss / total_batches if total_batches > 0 else 0,
        "rat_loss": total_rat_loss / total_batches if total_batches > 0 else 0,
        "sp_loss": total_sp_loss / total_batches if total_batches > 0 else 0,
        "acc_a": correct_a / total_samples if total_samples > 0 else 0,
        "acc_r": correct_r / total_samples if total_samples > 0 else 0,
        "acc_ar": correct_ar / total_samples if total_samples > 0 else 0,
        "time": epoch_time
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train CACR-SP Model")
    parser.add_argument("--data_dir", type=str, default="data/vcr", help="VCR data directory")
    parser.add_argument("--image_dir", type=str, default=None, help="Root directory containing vcr1images / movie folders.")
    parser.add_argument("--zip_path", type=str, default="data/vcr/vcr1images.zip", help="Zip path for images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.1, help="Scorer dropout")
    parser.add_argument("--ans_loss_weight", type=float, default=1.0, help="Weight for answer loss")
    parser.add_argument("--device", type=str, default="auto", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/cacr_sp", help="Save directory")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save mid-epoch checkpoint every N batches")
    parser.add_argument("--max_train_samples", type=int, default=-1, help="Subset for tiny overfit test")
    parser.add_argument("--max_val_samples", type=int, default=-1, help="Subset for validation")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # CACR-SP specific arguments
    parser.add_argument("--lambda_sp", type=float, default=0.1, help="Weight for shortcut penalty")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive loss")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--enable_sp", action="store_true", help="Enable shortcut penalty")

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_dataset = VCRDataset(split="train", data_dir=args.data_dir, image_dir=args.image_dir)
    val_dataset = VCRDataset(split="val", data_dir=args.data_dir, image_dir=args.image_dir)
    
    if args.max_train_samples > 0:
        indices = list(range(min(args.max_train_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        print(f"Subsampled train set to {len(train_dataset)} samples.")
        
    if args.max_val_samples > 0:
        indices = list(range(min(args.max_val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, indices)
        print(f"Subsampled val set to {len(val_dataset)} samples.")
        
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=vcr_collate_fn
    )
    
    print("Initializing models...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device.type)
    model = CACRSPVCRModel(vlm=vlm, scorer_dropout=args.dropout, embedding_dim=args.embedding_dim, temperature=args.temperature)
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(loss_type='infonce', temperature=args.temperature)
    sp_criterion = ShortcutPenalty(lambda_sp=args.lambda_sp, margin=0.25, formulation='confidence_penalty')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if not os.access(args.checkpoint_dir, os.W_OK):
        raise ValueError(f"Checkpoint directory {args.checkpoint_dir} is not writable.")
    
    start_epoch = 1
    best_val_acc = -1.0
    resume_batch_idx = 0
    metrics_state = None
    
    total_train_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        start_epoch = checkpoint.get("epoch", 1)
        resume_batch_idx = checkpoint.get("batch_idx", -1) + 1
        
        if resume_batch_idx >= total_train_batches:
            start_epoch += 1
            resume_batch_idx = 0
        else:
            metrics_state = checkpoint.get("metrics_state", None)
            if "torch_rng_state" in checkpoint:
                rng_state = checkpoint["torch_rng_state"]
                if isinstance(rng_state, torch.Tensor):
                    rng_state = rng_state.cpu()
                    if rng_state.dtype != torch.uint8:
                        rng_state = rng_state.type(torch.ByteTensor)
                torch.set_rng_state(rng_state)
        
        if "best_val_acc" in checkpoint:
            best_val_acc = checkpoint["best_val_acc"]
        elif "val_metrics" in checkpoint and "acc_ar" in checkpoint["val_metrics"]:
            best_val_acc = checkpoint["val_metrics"]["acc_ar"]
            
        print(f"Resuming epoch {start_epoch}")
        if resume_batch_idx > 0:
            print(f"Resuming from batch {resume_batch_idx + 1}/{total_train_batches}")
        print(f"Best Val Q->AR so far: {best_val_acc:.4f}")
        
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)
        indices = torch.randperm(len(train_dataset), generator=g).tolist()
        
        current_start_batch_idx = 0
        if epoch == start_epoch and resume_batch_idx > 0:
            current_start_batch_idx = resume_batch_idx
            start_sample_idx = resume_batch_idx * args.batch_size
            indices = indices[start_sample_idx:]
            
        epoch_dataset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            epoch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=vcr_collate_fn
        )
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, contrastive_criterion, sp_criterion, device, args,
            epoch=epoch, best_val_acc=best_val_acc, total_batches=total_train_batches,
            start_batch_idx=current_start_batch_idx, metrics_state=metrics_state
        )
        
        metrics_state = None
        
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Train Q->A Acc:  {train_metrics['acc_a']:.4f}")
        print(f"Train QA->R Acc: {train_metrics['acc_r']:.4f} (Teacher-forced)")
        print(f"Train Q->AR Acc: {train_metrics['acc_ar']:.4f} (Teacher-forced)")
        print(f"Time: {train_metrics['time']:.2f}s")
        
        print("\nRunning validation...")
        val_metrics = evaluate_cacr_sp(model, val_loader, device, zip_path=args.zip_path)
        
        print(f"Val Q->A Acc:  {val_metrics['acc_a']:.4f}")
        print(f"Val QA->R (TF): {val_metrics['acc_r_tf']:.4f}")
        print(f"Val QA->R (Pred): {val_metrics['acc_r_pred']:.4f}")
        print(f"Val Q->AR Acc: {val_metrics['acc_ar']:.4f}")
        print(f"Val Blind Acc: {val_metrics['acc_blind']:.4f}")
        
        if val_metrics["acc_ar"] > best_val_acc:
            best_val_acc = val_metrics["acc_ar"]
            save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            
            state_dict = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics
            }
            torch.save(state_dict, save_path)
            print(f"Saved best model with Val Q->AR: {best_val_acc:.4f} to {save_path}")

        temp_latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt.tmp")
        latest_save_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
        latest_state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "batch_idx": total_train_batches - 1,
            "best_val_acc": best_val_acc,
            "val_metrics": val_metrics,
            "torch_rng_state": torch.get_rng_state()
        }
        torch.save(latest_state_dict, temp_latest_path)
        os.replace(temp_latest_path, latest_save_path)
        print(f"Saved latest checkpoint: {latest_save_path}")

if __name__ == "__main__":
    main()
