import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image

@torch.no_grad()
def diagnose_detailed(model, dataloader, device, zip_path):
    model.eval()
    
    text_cos_sims = []
    text_mag_diffs = []
    text_l2_dists = []
    
    ctx_cos_sims = []
    ctx_mag_diffs = []
    ctx_l2_dists = []
    
    # Image dominance metrics
    ctx_diff_no_image = []
    ctx_diff_no_text = []
    ctx_diff_ans_change = []
    
    # Answer swap sensitivity
    ans_swap_ctx_cos_avg = []
    ans_swap_score_diff_avg = []
    changed_preds_ans_swap = 0
    total_ans_swaps = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        images = [load_image(p, zip_path) for p in batch["image_path"]]
        questions = batch["question"]
        answer_choices = batch["answer_choices"]
        ans_labels = batch["answer_label"].to(device)
        rationale_choices = batch["rationale_choices"]
        rat_labels = batch["rationale_label"].to(device)
        
        image_embs = model.encode_images(images) # (B, 768)
        
        # We need predicted answers
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_preds = ans_logits.argmax(dim=-1)
        
        gt_answers = []
        pred_answers = []
        for i, label_idx in enumerate(ans_labels):
            gt_answers.append(answer_choices[i][label_idx.item()])
            pred_answers.append(answer_choices[i][ans_preds[i].item()])
            
        result_tf = model.forward_rationale(images, questions, gt_answers, rationale_choices, image_embs=image_embs)
        result_pred = model.forward_rationale(images, questions, pred_answers, rationale_choices, image_embs=image_embs)
        
        for i in range(len(ans_labels)):
            if gt_answers[i] != pred_answers[i]:
                # 1. TEXT EMBEDDING DIFFERENCE
                t_tf = result_tf["context_text_embs"][i] # (768)
                t_pred = result_pred["context_text_embs"][i] # (768)
                
                text_cos = F.cosine_similarity(t_tf.unsqueeze(0), t_pred.unsqueeze(0)).item()
                text_l2 = torch.norm(t_tf - t_pred, p=2).item()
                text_mag = torch.mean(torch.abs(t_tf - t_pred)).item()
                
                text_cos_sims.append(text_cos)
                text_l2_dists.append(text_l2)
                text_mag_diffs.append(text_mag)
                
                # 2. CONTEXT PROJECTION DIFFERENCE
                c_tf = result_tf["context_emb"][i] # (512)
                c_pred = result_pred["context_emb"][i]
                
                ctx_cos = F.cosine_similarity(c_tf.unsqueeze(0), c_pred.unsqueeze(0)).item()
                ctx_l2 = torch.norm(c_tf - c_pred, p=2).item()
                ctx_mag = torch.mean(torch.abs(c_tf - c_pred)).item()
                
                ctx_cos_sims.append(ctx_cos)
                ctx_l2_dists.append(ctx_l2)
                ctx_mag_diffs.append(ctx_mag)
                
                ctx_diff_ans_change.append(ctx_l2)
                
                # 4. IMAGE DOMINANCE
                # Calculate for GT answer
                # h_context = torch.cat([image_embs, ctx_text_embs, image_embs * ctx_text_embs], dim=-1)
                img = image_embs[i:i+1] # (1, 768)
                txt = t_tf.unsqueeze(0) # (1, 768)
                zero_img = torch.zeros_like(img)
                zero_txt = torch.zeros_like(txt)
                
                h_base = torch.cat([img, txt, img * txt], dim=-1)
                ctx_base = model.context_projection(h_base)
                
                h_no_img = torch.cat([zero_img, txt, zero_img * txt], dim=-1)
                ctx_no_img = model.context_projection(h_no_img)
                
                h_no_txt = torch.cat([img, zero_txt, img * zero_txt], dim=-1)
                ctx_no_txt = model.context_projection(h_no_txt)
                
                ctx_diff_no_image.append(torch.norm(ctx_base - ctx_no_img, p=2).item())
                ctx_diff_no_text.append(torch.norm(ctx_base - ctx_no_txt, p=2).item())
                
                # 5. ANSWER-SWAP SENSITIVITY
                if len(ans_swap_ctx_cos_avg) < 100: # limit subset to 100
                    q = questions[i]
                    a_choices = answer_choices[i]
                    
                    r_texts = [f"Rationale: {r}" for r in rationale_choices[i]]
                    r_embs = model.vlm.encode_text(r_texts)
                    r_proj = model.rationale_projection(r_embs)
                    
                    all_ctx_embs = []
                    all_scores = []
                    for a in a_choices:
                        a_txt = f"Question: {q} Answer: {a}"
                        a_emb = model.vlm.encode_text([a_txt])
                        h = torch.cat([img, a_emb, img * a_emb], dim=-1)
                        c = model.context_projection(h)
                        all_ctx_embs.append(c)
                        sc = (r_proj * c).sum(-1)
                        all_scores.append(sc)
                        
                    # Compare pairs
                    pair_cos = []
                    pair_sc_diff = []
                    for x in range(4):
                        for y in range(x+1, 4):
                            pair_cos.append(F.cosine_similarity(all_ctx_embs[x], all_ctx_embs[y]).item())
                            pair_sc_diff.append(torch.mean(torch.abs(all_scores[x] - all_scores[y])).item())
                            
                    ans_swap_ctx_cos_avg.append(np.mean(pair_cos))
                    ans_swap_score_diff_avg.append(np.mean(pair_sc_diff))
                    
                    # Count changed rationale predictions across all 4 answers
                    preds = [s.argmax().item() for s in all_scores]
                    if len(set(preds)) > 1:
                        changed_preds_ans_swap += 1
                    total_ans_swaps += 1

    print("=== DIAGNOSTIC REPORT DETAILED ===")
    print(f"Total diff samples evaluated: {len(text_cos_sims)}")
    
    def report_stats(name, arr):
        print(f"{name}:")
        print(f"  Mean:   {np.mean(arr):.6f}")
        print(f"  Median: {np.median(arr):.6f}")
        print(f"  Std:    {np.std(arr):.6f}")
        print(f"  Min:    {np.min(arr):.6f}")
        print(f"  Max:    {np.max(arr):.6f}")
        
    print("\n1. TEXT EMBEDDING DIFFERENCE (RAW VLM)")
    report_stats("Cosine Sim", text_cos_sims)
    report_stats("Mean Abs Diff", text_mag_diffs)
    report_stats("L2 Dist", text_l2_dists)
    
    print("\n2. CONTEXT PROJECTION DIFFERENCE (FINAL)")
    report_stats("Cosine Sim", ctx_cos_sims)
    report_stats("Mean Abs Diff", ctx_mag_diffs)
    report_stats("L2 Dist", ctx_l2_dists)
    
    print("\n3. INFORMATION RETENTION")
    print(f"Avg L2 Change (Text): {np.mean(text_l2_dists):.6f} -> (Final Ctx): {np.mean(ctx_l2_dists):.6f}")
    print(f"Avg Cos Sim (Text): {np.mean(text_cos_sims):.6f} -> (Final Ctx): {np.mean(ctx_cos_sims):.6f}")
    
    print("\n4. IMAGE DOMINANCE (L2 distances from base context)")
    print(f"Avg diff when Answer Text changes (GT->Pred): {np.mean(ctx_diff_ans_change):.6f}")
    print(f"Avg diff when Image removed: {np.mean(ctx_diff_no_image):.6f}")
    print(f"Avg diff when Text removed: {np.mean(ctx_diff_no_text):.6f}")
    
    print("\n5. ANSWER-SWAP SENSITIVITY (Subset size: 100)")
    print(f"Avg pairwise context cosine sim: {np.mean(ans_swap_ctx_cos_avg):.6f}")
    print(f"Avg pairwise score diff: {np.mean(ans_swap_score_diff_avg):.6f}")
    print(f"Cases where answer swap changed selected rationale: {changed_preds_ans_swap} / {total_ans_swaps}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/vcr")
    parser.add_argument("--image_dir", type=str, default="data/vcr/vcr1images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = VCRDataset(split="val", data_dir=args.data_dir, image_dir=args.image_dir)
    indices = list(range(1000))
    dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=vcr_collate_fn)
    
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device.type)
    model = CACRSPVCRModel(vlm=vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07)
    model.to(device)
    
    checkpoint_path = "checkpoints/cacr_sp/latest_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
    
    diagnose_detailed(model, dataloader, device, args.image_dir)
    
if __name__ == "__main__":
    main()
