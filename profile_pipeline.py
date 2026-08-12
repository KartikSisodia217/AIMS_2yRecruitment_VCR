"""
Pipeline Profiler: Measures time spent in each stage of the VCR training pipeline.

Profiles:
- Image loading (from ZIP)
- Image embedding (VLM vision encoder)
- Text embedding (VLM text encoder)
- Fusion + Scoring (MLP)
- Total batch time

Also reports GPU/CPU info and memory usage.
"""

import os
import time
import torch
import platform

from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper
from src.model import BaselineVCRModel
from src.utils import load_image, set_seed


def profile_batch(model, images, questions, answer_choices, rationale_choices, ans_labels):
    """Profile a single batch through all stages."""
    timings = {}
    
    # --- Image Embedding ---
    t0 = time.perf_counter()
    image_embs = model.encode_images(images)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["image_embedding"] = time.perf_counter() - t0
    
    # --- Answer Text Embedding ---
    B = len(questions)
    ans_texts = []
    for q, a_list in zip(questions, answer_choices):
        for a in a_list:
            ans_texts.append(f"Question: {q} Answer: {a}")
    
    t0 = time.perf_counter()
    ans_text_embs = model.vlm.encode_text(ans_texts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["answer_text_embedding"] = time.perf_counter() - t0
    
    # --- Answer Scoring (MLP only) ---
    img_embs_expanded = image_embs.unsqueeze(1).expand(B, 4, -1).reshape(B * 4, -1)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        ans_scores = model.answer_scorer(img_embs_expanded, ans_text_embs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["answer_scoring"] = time.perf_counter() - t0
    
    ans_logits = ans_scores.view(B, 4)
    ans_preds = ans_logits.argmax(dim=-1)
    
    # --- Rationale Text Embedding ---
    selected_answers = []
    for i in range(B):
        selected_answers.append(answer_choices[i][ans_labels[i]])
    
    rat_texts = []
    for q, ans, r_list in zip(questions, selected_answers, rationale_choices):
        for r in r_list:
            rat_texts.append(f"Question: {q} Answer: {ans} Rationale: {r}")
    
    t0 = time.perf_counter()
    rat_text_embs = model.vlm.encode_text(rat_texts)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["rationale_text_embedding"] = time.perf_counter() - t0
    
    # --- Rationale Scoring (MLP only) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        rat_scores = model.rationale_scorer(img_embs_expanded, rat_text_embs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["rationale_scoring"] = time.perf_counter() - t0
    
    return timings


def main():
    set_seed(42)
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    # ============================================================
    # System Info
    # ============================================================
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        print("GPU: Not available (CPU mode)")
    
    # ============================================================
    # Load Components
    # ============================================================
    print("\n" + "=" * 60)
    print("LOADING COMPONENTS")
    print("=" * 60)
    
    dataset = VCRDataset(split='train', data_dir='data/vcr')
    zip_path = os.path.join("data", "vcr", "vcr1images.zip")
    
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device=device_str)
    model = BaselineVCRModel(vlm=vlm, scorer_dropout=0.1)
    model.to(device)
    model.eval()
    
    # ============================================================
    # Profile with batch_size=4, num_batches=4
    # ============================================================
    batch_size = 4
    num_batches = 4
    num_samples = batch_size * num_batches
    
    print(f"\nProfiling {num_batches} batches × {batch_size} samples = {num_samples} total")
    
    # Pre-load sample data
    all_samples = []
    for i in range(num_samples):
        all_samples.append(dataset[i])
    
    # Timing accumulators
    total_timings = {
        "image_loading": 0.0,
        "image_embedding": 0.0,
        "answer_text_embedding": 0.0,
        "answer_scoring": 0.0,
        "rationale_text_embedding": 0.0,
        "rationale_scoring": 0.0,
        "total_batch": 0.0,
    }
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_samples = all_samples[batch_start:batch_start + batch_size]
        
        # --- Image Loading ---
        t0 = time.perf_counter()
        images = [load_image(s["image_path"], zip_path) for s in batch_samples]
        total_timings["image_loading"] += time.perf_counter() - t0
        
        questions = [s["question"] for s in batch_samples]
        answer_choices = [s["answer_choices"] for s in batch_samples]
        rationale_choices = [s["rationale_choices"] for s in batch_samples]
        ans_labels = [s["answer_label"] for s in batch_samples]
        
        # --- Full batch profile ---
        t_batch = time.perf_counter()
        batch_timings = profile_batch(model, images, questions, answer_choices, rationale_choices, ans_labels)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_timings["total_batch"] += time.perf_counter() - t_batch
        
        for key in batch_timings:
            total_timings[key] += batch_timings[key]
    
    # ============================================================
    # Report
    # ============================================================
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"{'Stage':<30} {'Total (s)':<12} {'Per Batch (s)':<15} {'% of Batch':<10}")
    print("-" * 67)
    
    total_batch_time = total_timings["total_batch"] + total_timings["image_loading"]
    
    for key, label in [
        ("image_loading", "Image Loading (ZIP)"),
        ("image_embedding", "Image Embedding (VLM)"),
        ("answer_text_embedding", "Answer Text Embedding"),
        ("answer_scoring", "Answer Scoring (MLP)"),
        ("rationale_text_embedding", "Rationale Text Embedding"),
        ("rationale_scoring", "Rationale Scoring (MLP)"),
    ]:
        total = total_timings[key]
        per_batch = total / num_batches
        pct = (total / total_batch_time * 100) if total_batch_time > 0 else 0
        print(f"{label:<30} {total:<12.4f} {per_batch:<15.4f} {pct:<10.1f}%")
    
    print("-" * 67)
    print(f"{'Total (incl. loading)':<30} {total_batch_time:<12.4f} {total_batch_time/num_batches:<15.4f}")
    
    # Text embedding combined
    text_emb_total = total_timings["answer_text_embedding"] + total_timings["rationale_text_embedding"]
    scoring_total = total_timings["answer_scoring"] + total_timings["rationale_scoring"]
    
    print(f"\nSummary breakdown:")
    print(f"  Image loading:    {total_timings['image_loading']/total_batch_time*100:.1f}%")
    print(f"  Image embedding:  {total_timings['image_embedding']/total_batch_time*100:.1f}%")
    print(f"  Text embedding:   {text_emb_total/total_batch_time*100:.1f}%")
    print(f"  MLP scoring:      {scoring_total/total_batch_time*100:.1f}%")
    
    # ============================================================
    # Memory Report
    # ============================================================
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print("GPU MEMORY")
        print(f"{'='*60}")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # ============================================================
    # Bottleneck Analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*60}")
    
    stages = [
        ("Image Loading (ZIP)", total_timings["image_loading"]),
        ("Image Embedding", total_timings["image_embedding"]),
        ("Text Embedding (combined)", text_emb_total),
        ("MLP Scoring (combined)", scoring_total),
    ]
    stages.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Dominant bottleneck: {stages[0][0]} ({stages[0][1]/total_batch_time*100:.1f}%)")
    print(f"Second bottleneck:  {stages[1][0]} ({stages[1][1]/total_batch_time*100:.1f}%)")


if __name__ == "__main__":
    main()
