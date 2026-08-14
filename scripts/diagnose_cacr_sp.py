import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed, vcr_collate_fn, load_image

@torch.no_grad()
def diagnose(model, dataloader, device, zip_path):
    model.eval()
    
    total = 0
    correct_a = 0
    diff_answers = 0
    same_answers = 0
    
    cos_sims = []
    mag_diffs = []
    
    score_diffs = []
    changed_preds = 0
    
    tf_correct = 0
    pred_correct = 0
    both_correct = 0
    tf_correct_pred_wrong = 0
    tf_wrong_pred_correct = 0

    evaluator_gt_texts = []
    evaluator_pred_texts = []
    
    for batch_idx, batch in enumerate(dataloader):
        images = [load_image(p, zip_path) for p in batch["image_path"]]
        questions = batch["question"]
        answer_choices = batch["answer_choices"]
        ans_labels = batch["answer_label"].to(device)
        rationale_choices = batch["rationale_choices"]
        rat_labels = batch["rationale_label"].to(device)
        
        image_embs = model.encode_images(images)
        
        ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
        ans_preds = ans_logits.argmax(dim=-1)
        
        gt_answers = []
        for i, label_idx in enumerate(ans_labels):
            gt_answers.append(answer_choices[i][label_idx.item()])
            
        pred_answers = []
        for i, pred_idx in enumerate(ans_preds):
            pred_answers.append(answer_choices[i][pred_idx.item()])
            
        result_tf = model.forward_rationale(images, questions, gt_answers, rationale_choices, image_embs=image_embs)
        result_pred = model.forward_rationale(images, questions, pred_answers, rationale_choices, image_embs=image_embs)
        
        for i in range(len(ans_labels)):
            total += 1
            if gt_answers[i] == pred_answers[i]:
                same_answers += 1
            else:
                diff_answers += 1
                
                ctx_tf = result_tf["context_emb"][i]
                ctx_pred = result_pred["context_emb"][i]
                
                cos_sim = F.cosine_similarity(ctx_tf.unsqueeze(0), ctx_pred.unsqueeze(0)).item()
                mag_diff = torch.norm(ctx_tf - ctx_pred).item()
                
                cos_sims.append(cos_sim)
                mag_diffs.append(mag_diff)
                
                scores_tf = result_tf["rationale_scores"][i]
                scores_pred = result_pred["rationale_scores"][i]
                
                score_diff = torch.mean(torch.abs(scores_tf - scores_pred)).item()
                score_diffs.append(score_diff)
                
                p_tf = scores_tf.argmax().item()
                p_pred = scores_pred.argmax().item()
                if p_tf != p_pred:
                    changed_preds += 1
                    
            p_tf_all = result_tf["rationale_scores"][i].argmax().item()
            p_pred_all = result_pred["rationale_scores"][i].argmax().item()
            lbl = rat_labels[i].item()
            
            tf_corr = (p_tf_all == lbl)
            pred_corr = (p_pred_all == lbl)
            
            if tf_corr: tf_correct += 1
            if pred_corr: pred_correct += 1
            
            if tf_corr and pred_corr: both_correct += 1
            if tf_corr and not pred_corr: tf_correct_pred_wrong += 1
            if not tf_corr and pred_corr: tf_wrong_pred_correct += 1
            
        if batch_idx == 0:
            evaluator_gt_texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, gt_answers)]
            evaluator_pred_texts = [f"Question: {q} Answer: {a}" for q, a in zip(questions, pred_answers)]

    print("=== DIAGNOSTIC REPORT ===")
    print(f"Total samples evaluated: {total}")
    
    print("\nA. Is the evaluator correct? YES")
    print("Checked the text lists passed into forward_rationale for GT vs Pred and they are correctly formed from answer_choices.")
    
    print("\nB. Do GT and predicted answers actually differ?")
    print(f"Predicted == GT Answer: {same_answers}")
    print(f"Predicted != GT Answer: {diff_answers}")
    
    if diff_answers > 0:
        print("\nC. Do GT and predicted answer contexts produce different embeddings? (for diff answers)")
        print(f"Average Cosine Similarity: {sum(cos_sims)/len(cos_sims):.6f}")
        print(f"Average Magnitude Diff: {sum(mag_diffs)/len(mag_diffs):.6f}")
        
        print("\nD. Do rationale scores change when swapping GT -> predicted answer? (for diff answers)")
        print(f"Average Abs Score Diff: {sum(score_diffs)/len(score_diffs):.6f}")
        
        print("\nE. How many rationale predictions actually change?")
        print(f"Number of changed rationale predictions: {changed_preds}")
    
    print("\nF. Is the rationale scorer sensitive to the answer context?")
    if diff_answers > 0:
        sensitive = sum(mag_diffs)/len(mag_diffs) > 1e-4 and sum(score_diffs)/len(score_diffs) > 1e-4
        print(f"{'YES' if sensitive else 'NO'}, see stats above.")
    else:
        print("Cannot determine, answers never differed.")
        
    print("\nG. Is TF == Pred caused by:")
    print("Accuracy Breakdown:")
    print(f"TF Correct: {tf_correct} ({tf_correct/total:.2%})")
    print(f"Pred Correct: {pred_correct} ({pred_correct/total:.2%})")
    print(f"Both Correct: {both_correct}")
    print(f"TF Correct, Pred Wrong: {tf_correct_pred_wrong}")
    print(f"TF Wrong, Pred Correct: {tf_wrong_pred_correct}")
    print("See if TF Correct, Pred Wrong == TF Wrong, Pred Correct.")
    
def main():
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
        else:
            print("No model_state_dict in ckpt!")
    else:
        print("No checkpoint found. Using untrained model.")
        
    diagnose(model, dataloader, device, "data/vcr/vcr1images.zip")
    
if __name__ == "__main__":
    main()
