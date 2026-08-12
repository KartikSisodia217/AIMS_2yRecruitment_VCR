import os
import zipfile
import io
import torch
import torch.nn as nn
from PIL import Image

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.model import BaselineVCRModel

def get_image_from_zip(zip_path, img_path):
    img_path = img_path.replace("\\", "/")
    if "vcr1images/" in img_path:
        rel_path = "vcr1images/" + img_path.split("vcr1images/")[-1]
    else:
        raise ValueError(f"Cannot parse relative path from {img_path}")
        
    print(f"Reading {rel_path} from {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(rel_path) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")

def main():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    print("Loading VCRDataset...")
    dataset = VCRDataset(split='train', data_dir='data/vcr')
    sample = dataset[0]
    
    image_path = sample['image_path']
    zip_path = os.path.join("data", "vcr", "vcr1images.zip")
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist.")
        return
        
    image = get_image_from_zip(zip_path, image_path)
    
    question = sample['question']
    answer_choices = sample['answer_choices']
    answer_label = sample['answer_label']
    
    rationale_choices = sample['rationale_choices']
    rationale_label = sample['rationale_label']
    
    print("\nInitializing SigLIP2Wrapper (Frozen)...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device_str)
    
    print("Initializing BaselineVCRModel (Scorer Trainable)...")
    model = BaselineVCRModel(vlm=vlm, scorer_dropout=0.1)
    
    # Ensure entire model sits on the correct device
    model.to(device)
    
    # Parameter counts
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Parameter Counts ---")
    print(f"Frozen parameters (VLM): {frozen_params:,}")
    print(f"Trainable parameters (Scoring MLPs): {trainable_params:,}")
    
    # ----------------------------------------
    # Answer Selection
    # ----------------------------------------
    print("\n--- Answer Selection ---")
    
    images = [image]
    questions = [question]
    batch_answer_choices = [answer_choices]
    
    model.train()
    
    ans_logits = model.forward_answer(images, questions, batch_answer_choices)
    
    print(f"Answer Logits Shape: {ans_logits.shape}")
    print(f"Contains NaNs: {torch.isnan(ans_logits).any().item()}")
    
    criterion = nn.CrossEntropyLoss()
    ans_target = torch.tensor([answer_label], device=device)
    ans_loss = criterion(ans_logits, ans_target)
    
    print(f"Answer Loss: {ans_loss.item():.4f}")
    
    print("Backward pass on Answer Loss...")
    ans_loss.backward()
    
    has_grads_ans = any(p.grad is not None for p in model.answer_scorer.parameters())
    print(f"Produced gradients in Answer Scorer: {has_grads_ans}")
    
    # ----------------------------------------
    # Rationale Selection
    # ----------------------------------------
    print("\n--- Rationale Selection ---")
    # Using Ground Truth answer for rationale condition, as documented in src/model.py
    gt_answer = answer_choices[answer_label]
    selected_answers = [gt_answer]
    batch_rationale_choices = [rationale_choices]
    
    model.zero_grad()
    
    rat_logits = model.forward_rationale(images, questions, selected_answers, batch_rationale_choices)
    
    print(f"Rationale Logits Shape: {rat_logits.shape}")
    print(f"Contains NaNs: {torch.isnan(rat_logits).any().item()}")
    
    rat_target = torch.tensor([rationale_label], device=device)
    rat_loss = criterion(rat_logits, rat_target)
    
    print(f"Rationale Loss: {rat_loss.item():.4f}")
    
    print("Backward pass on Rationale Loss...")
    rat_loss.backward()
    
    has_grads_rat = any(p.grad is not None for p in model.rationale_scorer.parameters())
    print(f"Produced gradients in Rationale Scorer: {has_grads_rat}")

    # ----------------------------------------
    # VLM Gradient verification
    # ----------------------------------------
    vlm_has_grads = any(p.grad is not None for p in model.vlm.parameters())
    print(f"\nProduced gradients in VLM: {vlm_has_grads}")
    
    print("\nTest finished successfully!")

if __name__ == "__main__":
    main()
