import torch
from src.vlm import SigLIP2Wrapper
from src.model import BaselineVCRModel
from src.cacr_sp_model import CACRSPVCRModel
from src.utils import set_seed
import copy

def test_rng_isolation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock data
    batch_size = 2
    images = [torch.rand(3, 224, 224) for _ in range(batch_size)]  # Dummy images
    questions = ["q1", "q2"]
    answer_choices = [["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]]
    rationale_choices = [["r1", "r2", "r3", "r4"], ["s1", "s2", "s3", "s4"]]
    gt_answers = ["a1", "b1"]

    # 1. Initialize Baseline
    set_seed(42)
    vlm_base = SigLIP2Wrapper(device=device.type)
    model_base = BaselineVCRModel(vlm_base, scorer_dropout=0.1).to(device)
    model_base.train()
    
    # 2. Initialize CACR
    set_seed(42)
    vlm_cacr = SigLIP2Wrapper(device=device.type)
    model_cacr = CACRSPVCRModel(vlm_cacr, scorer_dropout=0.1).to(device)
    model_cacr.train()
    
    # Assert answer scorers are identically initialized
    for p1, p2 in zip(model_base.answer_scorer.parameters(), model_cacr.answer_scorer.parameters()):
        assert torch.allclose(p1, p2)

    # STEP 1
    # Base
    set_seed(42)
    start_rng_base = torch.get_rng_state()
    img_embs_base = model_base.encode_images(images)
    ans_logits_base = model_base.forward_answer(images, questions, answer_choices, image_embs=img_embs_base)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        rat_logits_base = model_base.forward_rationale(images, questions, gt_answers, rationale_choices, image_embs=img_embs_base)
    end_rng_base = torch.get_rng_state()
    
    # CACR
    set_seed(42)
    start_rng_cacr = torch.get_rng_state()
    img_embs_cacr = model_cacr.encode_images(images)
    ans_logits_cacr = model_cacr.forward_answer(images, questions, answer_choices, image_embs=img_embs_cacr)
    with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
        res_cacr = model_cacr.forward_rationale(images, questions, gt_answers, rationale_choices, image_embs=img_embs_cacr)
    end_rng_cacr = torch.get_rng_state()

    # Verify Step 1
    print("Step 1 Ans Logits Match:", torch.allclose(ans_logits_base, ans_logits_cacr))
    assert torch.allclose(ans_logits_base, ans_logits_cacr)
    print("RNG state preserved after rationale branch (Base):", torch.equal(start_rng_base, end_rng_base)) # False because ans_logits advances RNG!
    
    # Wait, ans_logits advances RNG, but forward_rationale should NOT advance it FURTHER.
    # To check this, compare end_rng_base and end_rng_cacr!
    print("End RNG Match (Base vs CACR):", torch.equal(end_rng_base, end_rng_cacr))
    assert torch.equal(end_rng_base, end_rng_cacr)

    # STEP 2
    # Base
    torch.set_rng_state(end_rng_base)
    img_embs_base_2 = model_base.encode_images(images)
    ans_logits_base_2 = model_base.forward_answer(images, questions, answer_choices, image_embs=img_embs_base_2)
    
    # CACR
    torch.set_rng_state(end_rng_cacr)
    img_embs_cacr_2 = model_cacr.encode_images(images)
    ans_logits_cacr_2 = model_cacr.forward_answer(images, questions, answer_choices, image_embs=img_embs_cacr_2)
    
    # Verify Step 2
    print("Step 2 Ans Logits Match:", torch.allclose(ans_logits_base_2, ans_logits_cacr_2))
    assert torch.allclose(ans_logits_base_2, ans_logits_cacr_2)

    print("SUCCESS: RNG isolation verified!")

if __name__ == "__main__":
    test_rng_isolation()
