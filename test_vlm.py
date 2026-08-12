import os
import zipfile
import io
import torch
from PIL import Image
from src.dataset import VCRDataset
from src.vlm import SigLIP2Wrapper

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
    print("Loading VCRDataset...")
    dataset = VCRDataset(split='train', data_dir='data/vcr')
    sample = dataset[0]
    
    image_path = sample['image_path']
    zip_path = os.path.join("data", "vcr", "vcr1images.zip")
    
    if not os.path.exists(zip_path):
        print(f"Error: {zip_path} does not exist.")
        return
        
    image = get_image_from_zip(zip_path, image_path)
    print(f"Loaded image size: {image.size}")
    
    question = sample['question']
    answer_candidate = sample['answer_choices'][0]
    text = f"Question: {question} Answer: {answer_candidate}"
    
    print(f"Text input: {text}")
    
    print("\nInitializing SigLIP2Wrapper...")
    vlm = SigLIP2Wrapper(model_name="google/siglip2-base-patch16-224", device="auto")
    
    print("\nRunning VLM forward pass...")
    outputs = vlm(images=[image], texts=[text])
    
    img_embed = outputs["image_embeds"]
    text_embed = outputs["text_embeds"]
    
    print("\n--- Image Representation ---")
    print(f"Shape: {img_embed.shape}")
    print(f"Dtype: {img_embed.dtype}")
    print(f"Device: {img_embed.device}")
    print(f"Contains NaN: {torch.isnan(img_embed).any().item()}")
    
    print("\n--- Text Representation ---")
    print(f"Shape: {text_embed.shape}")
    print(f"Dtype: {text_embed.dtype}")
    print(f"Device: {text_embed.device}")
    print(f"Contains NaN: {torch.isnan(text_embed).any().item()}")
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
