"""Utility to inspect VCR data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.debug_dataset import DebugVCRDataset

def main():
    print("Inspecting VCR Debug Dataset...")
    dataset = DebugVCRDataset(num_samples=5, create_images=False)
    print(f"Sample count: {len(dataset)}")
    
    for i in range(min(3, len(dataset))):
        s = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"ID: {s.sample_id}")
        print(f"Question: {s.question}")
        print(f"Answers: {len(s.answer_choices)}")
        print(f"Rationales: {len(s.rationale_choices)}")
        print(f"Answer Label: {s.answer_label}")
        print(f"Rationale Label: {s.rationale_label}")
        print(f"Is Valid: {s.is_valid}")

if __name__ == '__main__':
    main()
