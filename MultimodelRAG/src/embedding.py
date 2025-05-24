from langchain.embeddings import HuggingFaceEmbeddings
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os
from src.config import IMAGE_EMBEDDING_MODEL

class TextEmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

class ImageEmbeddingModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or IMAGE_EMBEDDING_MODEL
        
        # Load CLIP model for image embeddings
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    def get_image_embedding(self, image_path):
        """Generate embedding for a single image"""
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
            # Convert to numpy array and normalize
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / embedding.norm()
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {str(e)}")
            return None
    
    def embeddings(self, image_chunks):
        """Generate embeddings for multiple image chunks"""
        embeddings = []
        
        for chunk in image_chunks:
            image_path = chunk["image_path"]
            embedding = self.get_image_embedding(image_path)
            
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Use a zero vector as fallback
                embeddings.append(torch.zeros(self.model.config.projection_dim).numpy())
        
        return embeddings