import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class SigLIP2Wrapper(nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224", device="auto"):
        """
        Loads the pretrained SigLIP2 model and processor.
        :param model_name: Hugging Face model identifier
        :param device: 'cuda', 'cpu', or 'auto'
        """
        super().__init__()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def forward(self, images, texts):
        """
        Extracts representations for images and texts (joint forward).
        
        :param images: PIL Image or list of PIL Images
        :param texts: string or list of strings
        :return: dict with 'image_embeds' and 'text_embeds' (shapes [batch_size, embed_dim])
        
        NOTE: This jointly processes images and texts. Each image is paired 1:1 with a text.
        For efficiency with frozen VLM, prefer encode_image() + encode_text() to avoid
        re-encoding the same image multiple times.
        """
        # Process inputs
        inputs = self.processor(
            text=texts, 
            images=images, 
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # We only want representations, no training of the VLM yet
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return {
            "image_embeds": outputs.image_embeds,
            "text_embeds": outputs.text_embeds,
        }
    
    def encode_image(self, images):
        """
        Encodes only images, independently of text.
        Produces the same L2-normalized embeddings as forward().image_embeds.
        
        :param images: PIL Image or list of PIL Images
        :return: tensor of shape [batch_size, embed_dim] (L2-normalized)
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_outputs = self.model.get_image_features(**inputs)
        # get_image_features returns BaseModelOutputWithPooling; extract pooler_output
        pooler_output = vision_outputs.pooler_output
        # L2-normalize to match the joint forward() behavior exactly
        image_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return image_embeds
    
    def encode_text(self, texts):
        """
        Encodes only texts, independently of images.
        Produces the same L2-normalized embeddings as forward().text_embeds.
        
        :param texts: string or list of strings
        :return: tensor of shape [batch_size, embed_dim] (L2-normalized)
        """
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs)
        # get_text_features returns BaseModelOutputWithPooling; extract pooler_output
        pooler_output = text_outputs.pooler_output
        # L2-normalize to match the joint forward() behavior exactly
        text_embeds = pooler_output / pooler_output.norm(p=2, dim=-1, keepdim=True)
        return text_embeds
