import os
from src.dataset import VCRDataset

def main():
    print("Loading VCRDataset...")
    dataset = VCRDataset(split='train', data_dir='data/vcr')
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    
    print("\n--- Sample 0 ---")
    image_path = sample['image_path']
    metadata_path = sample['metadata_path']
    
    print(f"Image path: {image_path}")
    print(f"Image exists: {os.path.exists(image_path)}")
    print(f"Metadata path: {metadata_path}")
    print(f"Metadata exists: {os.path.exists(metadata_path)}")
    
    print("\nObjects:")
    print(sample['objects'])
    
    print("\nQuestion:")
    print(sample['question'])
    
    print("\nAnswer Choices:")
    for i, choice in enumerate(sample['answer_choices']):
        print(f"{i}: {choice}")
        
    print(f"\nAnswer Label: {sample['answer_label']}")
    
    print("\nRationale Choices:")
    for i, choice in enumerate(sample['rationale_choices']):
        print(f"{i}: {choice}")
        
    print(f"\nRationale Label: {sample['rationale_label']}")
    
    if os.path.exists(metadata_path):
        print("\nLoading Metadata...")
        meta = dataset.load_metadata(metadata_path)
        print("Metadata Keys:", list(meta.keys()))
        print(f"Width: {meta['width']}, Height: {meta['height']}")
        print(f"Number of boxes: {len(meta['boxes'])}")
        print(f"Number of segms: {len(meta['segms'])}")
    else:
        print("\nMetadata file does not exist. (Remember vcr1images.zip might not be extracted yet)")

if __name__ == '__main__':
    main()
