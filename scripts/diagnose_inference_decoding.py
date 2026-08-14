import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

def diagnose_decoding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    zip_path = "data/vcr/vcr1images.zip"
    data_dir = "data/vcr"
    
    val_dataset = VCRDataset(split="val", data_dir=data_dir)
    # Subset to 512 samples
    max_val_samples = 512
    indices = list(range(min(max_val_samples, len(val_dataset))))
    val_dataset = Subset(val_dataset, indices)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0,
        collate_fn=vcr_collate_fn
    )
    
    print("Initializing model...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device.type)
    model = CACRSPVCRModel(vlm=vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07)
    
    ckpt_path = "checkpoints/cacr_sp/latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/cacr_sp/best_model.pt"
    
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Patch model architecture if checkpoint uses MLP
    if "context_projection.mlp.0.weight" in state_dict:
        w0 = state_dict["context_projection.mlp.0.weight"]
        w2 = state_dict["context_projection.mlp.2.weight"]
        model.context_projection = MLP(w0.shape[1], w0.shape[0], w2.shape[0])
        print(f"Patched context_projection to MLP({w0.shape[1]}, {w0.shape[0]}, {w2.shape[0]})")
        
    if "rationale_projection.mlp.0.weight" in state_dict:
        w0 = state_dict["rationale_projection.mlp.0.weight"]
        w2 = state_dict["rationale_projection.mlp.2.weight"]
        model.rationale_projection = MLP(w0.shape[1], w0.shape[0], w2.shape[0])
        print(f"Patched rationale_projection to MLP({w0.shape[1]}, {w0.shape[0]}, {w2.shape[0]})")
        
    # Ignore keys that are missing if we didn't patch them, or just use strict=False
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Metrics
    total = 0
    correct_r_tf = 0
    correct_r_pred = 0
    
    # Formulation A
    correct_ar_A = 0
    correct_ans_joint_A = 0
    
    # Formulation B
    correct_ar_B = 0
    correct_ans_joint_B = 0
    
    changed_predictions = 0
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(val_loader)}")
                
            images = [load_image(p, zip_path) for p in batch["image_path"]]
            questions = batch["question"]
            answer_choices = batch["answer_choices"]
            ans_labels = batch["answer_label"].to(device)
            rationale_choices = batch["rationale_choices"]
            rat_labels = batch["rationale_label"].to(device)
            
            image_embs = model.encode_images(images)
            ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
            ans_preds = ans_logits.argmax(dim=-1)
            
            result_joint = model.forward_joint_rationale(
                images, questions, answer_choices, rationale_choices, image_embs=image_embs
            )
            joint_rat_scores = result_joint["joint_rationale_scores"]
            B = len(ans_labels)
            
            # QA->R TF
            rat_scores_tf = joint_rat_scores[torch.arange(B), ans_labels, :]
            correct_r_tf += (rat_scores_tf.argmax(dim=-1) == rat_labels).sum().item()
            
            # QA->R Pred
            rat_scores_pred = joint_rat_scores[torch.arange(B), ans_preds, :]
            correct_r_pred += (rat_scores_pred.argmax(dim=-1) == rat_labels).sum().item()
            
            temperature = getattr(model, 'temperature', 0.07)
            scaled_rat_scores = joint_rat_scores / temperature
            
            # Formulation A: Raw addition
            joint_scores_A = ans_logits.unsqueeze(2) + scaled_rat_scores
            joint_preds_flat_A = joint_scores_A.view(B, 16).argmax(-1)
            ans_preds_A = joint_preds_flat_A // 4
            rat_preds_A = joint_preds_flat_A % 4
            
            correct_ar_A += ((ans_preds_A == ans_labels) & (rat_preds_A == rat_labels)).sum().item()
            correct_ans_joint_A += (ans_preds_A == ans_labels).sum().item()
            
            # Formulation B: Hierarchical log-probability
            ans_logprobs = F.log_softmax(ans_logits, dim=-1)
            rat_logprobs = F.log_softmax(scaled_rat_scores, dim=-1)
            joint_scores_B = ans_logprobs.unsqueeze(2) + rat_logprobs
            joint_preds_flat_B = joint_scores_B.view(B, 16).argmax(-1)
            ans_preds_B = joint_preds_flat_B // 4
            rat_preds_B = joint_preds_flat_B % 4
            
            correct_ar_B += ((ans_preds_B == ans_labels) & (rat_preds_B == rat_labels)).sum().item()
            correct_ans_joint_B += (ans_preds_B == ans_labels).sum().item()
            
            changed_predictions += (joint_preds_flat_A != joint_preds_flat_B).sum().item()
            
            total += B

    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"QA->R TF:   {correct_r_tf / total * 100:.2f}%")
    print(f"QA->R Pred: {correct_r_pred / total * 100:.2f}%")
    print("\nFORMULATION A (Raw Addition - Current)")
    print(f"Q->AR:               {correct_ar_A / total * 100:.2f}%")
    print(f"Joint-decoded Q->A:  {correct_ans_joint_A / total * 100:.2f}%")
    
    print("\nFORMULATION B (Hierarchical Log-Softmax)")
    print(f"Q->AR:               {correct_ar_B / total * 100:.2f}%")
    print(f"Joint-decoded Q->A:  {correct_ans_joint_B / total * 100:.2f}%")
    
    print(f"\nChanged Predictions: {changed_predictions} / {total} ({(changed_predictions/total)*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    diagnose_decoding()
