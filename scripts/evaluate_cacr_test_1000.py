import os
import torch
from torch.utils.data import DataLoader, Subset
from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import vcr_collate_fn, load_image

def evaluate_cacr_test_1000():
    # Configuration
    DATA_DIR = "/kaggle/input/datasets/kartiksisodia217/aims-vcr-recruitment-dataset"
    IMAGE_DIR = "/kaggle/input/datasets/kartiksisodia217/aims-vcr-recruitment-dataset/vcr1images_clean"
    CHECKPOINT = "checkpoints/cacr_sp/epoch1_40_62_checkpoint.pt"
    SPLIT = "test"
    MAX_TEST_SAMPLES = 1000
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda")

    print("=== TEST SANITY CHECK CONFIGURATION ===")
    print(f"Split: {SPLIT}")
    print(f"Max Samples: {MAX_TEST_SAMPLES}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")
    print("=======================================\n")

    print(f"Loading {SPLIT} dataset...")
    # NOTE: The dataset implicitly uses test.jsonl from DATA_DIR / vcr1annots
    test_dataset = VCRDataset(split=SPLIT, data_dir=DATA_DIR, image_dir=IMAGE_DIR)
    
    # Restrict to exactly 1,000 samples
    num_loaded = len(test_dataset)
    print(f"Total samples found in {SPLIT} split: {num_loaded}")
    
    if num_loaded < MAX_TEST_SAMPLES:
        raise ValueError(f"Not enough samples in test set. Found {num_loaded}, needed {MAX_TEST_SAMPLES}.")
        
    indices = list(range(MAX_TEST_SAMPLES))
    test_dataset = Subset(test_dataset, indices)
    print(f"Actual samples loaded for evaluation: {len(test_dataset)}")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=vcr_collate_fn,
        num_workers=0
    )

    print("\nInitializing model...")
    # Using existing architecture
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=DEVICE.type)
    model = CACRSPVCRModel(vlm=vlm, scorer_dropout=0.1, embedding_dim=512, temperature=0.07)
    model.to(DEVICE)

    print(f"Loading checkpoint from {CHECKPOINT}...")
    checkpoint_state = torch.load(CHECKPOINT, map_location=DEVICE)
    if "model_state_dict" in checkpoint_state:
        model.load_state_dict(checkpoint_state["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_state)
        
    model.eval()

    print("\nStarting evaluation...")
    total = 0
    correct_a = 0
    correct_r_tf = 0
    correct_r_pred = 0
    correct_ar = 0
    correct_blind = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            images = [load_image(p, zip_path=None) for p in batch["image_path"]]
            questions = batch["question"]
            answer_choices = batch["answer_choices"]
            ans_labels = batch["answer_label"].to(DEVICE)
            rationale_choices = batch["rationale_choices"]
            rat_labels = batch["rationale_label"].to(DEVICE)
            
            # 1. Encode images ONCE
            image_embs = model.encode_images(images)
            
            # 2. Predict Answer
            ans_logits = model.forward_answer(images, questions, answer_choices, image_embs=image_embs)
            ans_preds = ans_logits.argmax(dim=-1)
            match_a = (ans_preds == ans_labels)
            correct_a += match_a.sum().item()
            
            # 3. Joint inference
            result_joint = model.forward_joint_rationale(
                images, questions, answer_choices, rationale_choices, image_embs=image_embs
            )
            
            B = len(ans_labels)
            joint_rat_scores = result_joint["joint_rationale_scores"]
            
            # QA->R (Teacher-Forced)
            rat_scores_tf = joint_rat_scores[torch.arange(B), ans_labels, :]
            rat_preds_tf = rat_scores_tf.argmax(dim=-1)
            correct_r_tf += (rat_preds_tf == rat_labels).sum().item()
            
            # 4. QA->R (Predicted Answer)
            rat_scores_pred = joint_rat_scores[torch.arange(B), ans_preds, :]
            rat_preds_pred = rat_scores_pred.argmax(dim=-1)
            match_r_pred = (rat_preds_pred == rat_labels)
            correct_r_pred += match_r_pred.sum().item()
            
            # 5. Q->AR (Joint Inference)
            temperature = getattr(model, 'temperature', 0.07)
            
            # User requirement: Do NOT use log-softmax.
            # joint_scores = ans_logits.unsqueeze(2) + joint_rationale_scores / temperature
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
            
            total += B
            
            # Print progress every 100 samples
            if total % 100 == 0 or total == MAX_TEST_SAMPLES:
                print(f"Processed {total}/{MAX_TEST_SAMPLES} samples...")

    # Calculate final metrics
    metrics = {
        "acc_a": correct_a / total if total > 0 else 0.0,
        "acc_r_tf": correct_r_tf / total if total > 0 else 0.0,
        "acc_r_pred": correct_r_pred / total if total > 0 else 0.0,
        "acc_ar": correct_ar / total if total > 0 else 0.0,
        "acc_blind": correct_blind / total if total > 0 else 0.0,
    }

    print("\n" + "="*40)
    print("          TEST SANITY CHECK          ")
    print("="*40)
    print(f"Test Q->A Acc:     {metrics['acc_a']:.4f}")
    print(f"Test QA->R (TF):   {metrics['acc_r_tf']:.4f}")
    print(f"Test QA->R (Pred): {metrics['acc_r_pred']:.4f}")
    print(f"Test Q->AR Acc:    {metrics['acc_ar']:.4f}")
    print(f"Test Blind Acc:    {metrics['acc_blind']:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_cacr_test_1000()
