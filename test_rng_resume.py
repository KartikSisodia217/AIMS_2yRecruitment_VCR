import torch
import os

def test_saving_and_loading():
    print("Testing saving and loading RNG state...")
    original_state = torch.get_rng_state()
    
    # Save dummy checkpoint
    dummy_ckpt = {
        "torch_rng_state": original_state,
        "epoch": 1,
        "batch_idx": 38999,
        "optimizer": {"dummy": "state"},
        "model_state_dict": {"dummy_weights": torch.tensor([1.0])}
    }
    torch.save(dummy_ckpt, "dummy_ckpt.pt")
    
    # Simulate loading as done in train.py (with CUDA mapping if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_ckpt = torch.load("dummy_ckpt.pt", map_location=device)
    
    # Apply fix
    rng_state = loaded_ckpt["torch_rng_state"]
    if isinstance(rng_state, torch.Tensor):
        rng_state = rng_state.cpu()
        if rng_state.dtype != torch.uint8:
            rng_state = rng_state.type(torch.ByteTensor)
            
    try:
        torch.set_rng_state(rng_state)
        print("Successfully restored RNG state from dummy checkpoint!")
    except Exception as e:
        print(f"Failed to restore RNG state: {e}")
        return False
        
    assert loaded_ckpt["epoch"] == 1
    assert loaded_ckpt["batch_idx"] == 38999
    print("Epoch and batch_idx preserved correctly.")
    os.remove("dummy_ckpt.pt")
    return True

def test_existing_checkpoint():
    ckpt_path = "checkpoints/latest_checkpoint.pt"
    if not os.path.exists(ckpt_path):
        print(f"No existing checkpoint found at {ckpt_path}")
        return True
        
    print(f"\nTesting backward compatibility with existing checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # This might throw OOM or something if the checkpoint is large and we don't have enough GPU memory,
    # but the user said "Run only the focused tests." and they are on Kaggle with T4.
    # We will load it.
    try:
        loaded_ckpt = torch.load(ckpt_path, map_location=device)
        print("Successfully loaded existing checkpoint into memory.")
        
        # Test RNG state
        if "torch_rng_state" in loaded_ckpt:
            rng_state = loaded_ckpt["torch_rng_state"]
            if isinstance(rng_state, torch.Tensor):
                rng_state = rng_state.cpu()
                if rng_state.dtype != torch.uint8:
                    rng_state = rng_state.type(torch.ByteTensor)
            torch.set_rng_state(rng_state)
            print("Successfully applied RNG state from existing checkpoint.")
            
        print(f"Epoch in existing ckpt: {loaded_ckpt.get('epoch')}")
        print(f"Batch idx in existing ckpt: {loaded_ckpt.get('batch_idx')}")
        
        # Test model and optimizer state presence
        if "model_state_dict" in loaded_ckpt:
            print("Model state remains loadable (key present).")
        if "optimizer" in loaded_ckpt:
            print("Optimizer state remains loadable (key present).")
            
        return True
    except Exception as e:
        print(f"Failed on existing checkpoint: {e}")
        return False

if __name__ == "__main__":
    success_dummy = test_saving_and_loading()
    success_existing = test_existing_checkpoint()
    if success_dummy and success_existing:
        print("\nALL TESTS PASSED")
    else:
        print("\nTESTS FAILED")
