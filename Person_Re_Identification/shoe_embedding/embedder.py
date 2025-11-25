import cv2
import torch
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class ShoeEmbedder:
    """
    A class to generate embeddings from shoe images using a pretrained ResNet-50 model.
    Handles image preprocessing, mirroring, and embedding averaging.
    """

    def __init__(self):
        """
        Initializes the ShoeEmbedder.
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # print(f"Using device for shoe embedding: {self.device}")

        # Load pretrained ResNet-50
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        base_model = resnet50(weights=self.weights)
        
        # Remove the final classification layer to get 2048-dim feature embeddings
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        # Get the appropriate preprocessing function
        self.preprocess = self.weights.transforms()

    def _get_embedding(self, image):
        """
        Generates a feature embedding for a single image.

        Args:
            image (PIL.Image): The input shoe image in PIL format.

        Returns:
            np.ndarray: The feature embedding.
        """
        # Preprocess the image
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            
        return embedding.cpu().numpy().flatten()

    def create_fused_embedding(self, shoe_crop):
        """
        Creates a fused embedding from an original shoe crop and its mirrored version.

        Args:
            shoe_crop (np.ndarray): The cropped image of a shoe.

        Returns:
            np.ndarray: The averaged embedding of the original and mirrored shoe.
                        Returns None if the crop is invalid.
        """
        if shoe_crop is None or shoe_crop.size == 0:
            return None
        
        # 1. Original shoe embedding
        # Convert NumPy array (OpenCV BGR) to PIL Image (RGB)
        original_rgb = cv2.cvtColor(shoe_crop, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_embedding = self._get_embedding(original_pil)
        
        # 2. Mirrored shoe embedding
        mirrored_shoe = cv2.flip(shoe_crop, 1)
        mirrored_rgb = cv2.cvtColor(mirrored_shoe, cv2.COLOR_BGR2RGB)
        mirrored_pil = Image.fromarray(mirrored_rgb)
        mirrored_embedding = self._get_embedding(mirrored_pil)
        
        # 3. Average the embeddings
        fused_embedding = np.mean([original_embedding, mirrored_embedding], axis=0)
        
        # Normalize the shoe embedding to unit length
        embedding_norm = np.linalg.norm(fused_embedding)
        if embedding_norm > 0:
            fused_embedding = fused_embedding / embedding_norm
        
        return fused_embedding 