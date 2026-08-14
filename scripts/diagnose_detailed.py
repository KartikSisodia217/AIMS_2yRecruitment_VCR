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
    
    # Text and Final Context diffs (pred vs gt)
    text_cos_sims = []
    text_l2_dists = []
    
    ctx_cos_sims = []
    ctx_l2_dists = []
    
    # Image dominance metrics
    ctx_diff_no_image = []
    ctx_diff_no_interaction = []
    
    # Answer-swap pairwise metrics
    ans_swap_raw_text_cos_avg = []
    ans_swap_int_proj_cos_avg = []
    ans_swap_txt_res_cos_avg = []
    ans_swap_final_ctx_cos_avg = []
    ans_swap_score_diff_avg = []
    ans_swap_variance_avg = []
    
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
            img = image_embs[i:i+1] # (1, 768)
            t_tf = result_tf["context_text_embs"][i:i+1] # (1, 768)
            
            # Modality Ablations (Clean isolation)
            # Full context
            ctx_base = model.compute_context(img, t_tf)
            
            # No Image -> isolating text residual normalized
            txt_res_unnorm = model.text_residual_projection(t_tf)
            ctx_no_img = F.normalize(txt_res_unnorm, p=2, dim=-1)
            
            # No text residual -> isolating interaction normalized
            concat_features = torch.cat([img, t_tf, img * t_tf], dim=-1)
            int_proj_unnorm = model.context_projection(concat_features)
            ctx_no_interaction = F.normalize(int_proj_unnorm, p=2, dim=-1)
            
            ctx_diff_no_image.append(torch.norm(ctx_base - ctx_no_img, p=2).item())
            ctx_diff_no_interaction.append(torch.norm(ctx_base - ctx_no_interaction, p=2).item())
            
            if gt_answers[i] != pred_answers[i]:
                # 1. TEXT EMBEDDING DIFFERENCE
                t_tf_s = result_tf["context_text_embs"][i] # (768)
                t_pred_s = result_pred["context_text_embs"][i] # (768)
                
                text_cos = F.cosine_similarity(t_tf_s.unsqueeze(0), t_pred_s.unsqueeze(0)).item()
                text_l2 = torch.norm(t_tf_s - t_pred_s, p=2).item()
                text_cos_sims.append(text_cos)
                text_l2_dists.append(text_l2)
                
                # 2. CONTEXT PROJECTION DIFFERENCE
                c_tf = result_tf["context_emb"][i] # (512)
                c_pred = result_pred["context_emb"][i]
                
                ctx_cos = F.cosine_similarity(c_tf.unsqueeze(0), c_pred.unsqueeze(0)).item()
                ctx_l2 = torch.norm(c_tf - c_pred, p=2).item()
                
                ctx_cos_sims.append(ctx_cos)
                ctx_l2_dists.append(ctx_l2)
                
            # ANSWER-SWAP SENSITIVITY (for all, max 100 samples to keep it fast)
            if len(ans_swap_final_ctx_cos_avg) < 100:
                q = questions[i]
                a_choices = answer_choices[i]
                
                r_texts = [f"Rationale: {r}" for r in rationale_choices[i]]
                r_embs = model.vlm.encode_text(r_texts)
                r_proj = F.normalize(model.rationale_projection(r_embs), p=2, dim=-1)
                
                all_raw_text = []
                all_int_proj = []
                all_txt_res = []
                all_final_ctx = []
                all_scores = []
                
                for a in a_choices:
                    a_txt = f"Question: {q} Answer: {a}"
                    a_emb = model.vlm.encode_text([a_txt]) # (1, 768)
                    
                    # Store raw text
                    all_raw_text.append(a_emb)
                    
                    # Store interaction proj
                    concat_feat = torch.cat([img, a_emb, img * a_emb], dim=-1)
                    int_p = model.context_projection(concat_feat)
                    all_int_proj.append(int_p)
                    
                    # Store text residual
                    txt_p = model.text_residual_projection(a_emb)
                    all_txt_res.append(txt_p)
                    
                    # Store final context
                    c = model.compute_context(img, a_emb)
                    all_final_ctx.append(c)
                    
                    # Score
                    sc = (r_proj * c).sum(-1)
                    all_scores.append(sc)
                    
                # Pairwise similarities
                def avg_pairwise_cos(embs):
                    cos_list = []
                    for x in range(4):
                        for y in range(x+1, 4):
                            cos_list.append(F.cosine_similarity(embs[x], embs[y]).item())
                    return np.mean(cos_list)
                    
                ans_swap_raw_text_cos_avg.append(avg_pairwise_cos(all_raw_text))
                ans_swap_int_proj_cos_avg.append(avg_pairwise_cos(all_int_proj))
                ans_swap_txt_res_cos_avg.append(avg_pairwise_cos(all_txt_res))
                ans_swap_final_ctx_cos_avg.append(avg_pairwise_cos(all_final_ctx))
                
                # Pairwise score difference
                pair_sc_diff = []
                for x in range(4):
                    for y in range(x+1, 4):
                        pair_sc_diff.append(torch.mean(torch.abs(all_scores[x] - all_scores[y])).item())
                ans_swap_score_diff_avg.append(np.mean(pair_sc_diff))
                
                # Answer variance measure: mean_i ||C_i - mean(C)||²
                ctx_stack = torch.cat(all_final_ctx, dim=0) # (4, 512)
                ctx_mean = ctx_stack.mean(dim=0, keepdim=True) # (1, 512)
                var = torch.norm(ctx_stack - ctx_mean, p=2, dim=-1).pow(2).mean().item()
                ans_swap_variance_avg.append(var)
                
                # Count changed rationale predictions across all 4 answers
                preds = [s.argmax().item() for s in all_scores]
                if len(set(preds)) > 1:
                    changed_preds_ans_swap += 1
                total_ans_swaps += 1

    print("=== DIAGNOSTIC REPORT DETAILED ===")
    
    def report_stats(name, arr):
        print(f"{name}:")
        if len(arr) == 0:
            print("  No samples evaluated.")
            return
        print(f"  Mean:   {np.mean(arr):.6f}")
        print(f"  Median: {np.median(arr):.6f}")
        print(f"  Std:    {np.std(arr):.6f}")
        print(f"  Min:    {np.min(arr):.6f}")
        print(f"  Max:    {np.max(arr):.6f}")
        
    print(f"\nTotal diff samples evaluated (pred != gt): {len(text_cos_sims)}")
    print("\n1. TEXT EMBEDDING DIFFERENCE (RAW VLM)")
    report_stats("Cosine Sim", text_cos_sims)
    report_stats("L2 Dist", text_l2_dists)
    
    print("\n2. CONTEXT PROJECTION DIFFERENCE (FINAL)")
    report_stats("Cosine Sim", ctx_cos_sims)
    report_stats("L2 Dist", ctx_l2_dists)
    
    print("\n3. IMAGE DOMINANCE (L2 distances from base context)")
    print(f"Avg diff when Image removed (Text Residual only): {np.mean(ctx_diff_no_image):.6f}")
    print(f"Avg diff when Text Residual removed (Interaction only): {np.mean(ctx_diff_no_interaction):.6f}")
    
    print("\n4. ANSWER-SWAP SENSITIVITY (Subset size: 100)")
    print("Pairwise Cosine Similarities (Lower means more answer variance):")
    print(f"  A. Raw text representation:       {np.mean(ans_swap_raw_text_cos_avg):.6f}")
    print(f"  B. Original interaction proj:     {np.mean(ans_swap_int_proj_cos_avg):.6f}")
    print(f"  C. Text residual proj:            {np.mean(ans_swap_txt_res_cos_avg):.6f}")
    print(f"  D. Final Context:                 {np.mean(ans_swap_final_ctx_cos_avg):.6f}")
    
    print(f"\n  E. Answer Variance (mean ||C_i - mean(C)||²): {np.mean(ans_swap_variance_avg):.6f}")
    print(f"  Avg pairwise score diff: {np.mean(ans_swap_score_diff_avg):.6f}")
    print(f"  Cases where answer swap changed selected rationale: {changed_preds_ans_swap} / {total_ans_swaps}")

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
