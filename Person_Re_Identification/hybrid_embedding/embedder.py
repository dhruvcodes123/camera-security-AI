import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional

# Add project root to sys.path to allow for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from gait_extraction.extractor import GaitExtractor

class GaitEmbedder:
    """
    A wrapper for the GaitExtractor to produce final gait embeddings.
    """
    def __init__(self, embedding_dim=1024):
        """
        Initializes the gait embedding extractor.
        """
        print("Initializing GaitEmbedder...")
        self.gait_extractor = GaitExtractor()
        
        # This layer projects the raw gait features into the final embedding space.
        self.gait_input_dim = 3840  # Real SwinGait outputs 3840D embeddings
        self.gait_projection = nn.Linear(self.gait_input_dim, embedding_dim)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gait_projection.to(self.device)

        print("GaitEmbedder initialized successfully.")

    def get_embedding(self, image_sequence: list) -> Optional[Tuple[np.ndarray, str, list]]:
        """
        Extracts a normalized gait embedding from a sequence of images.
        """
        if not image_sequence:
            return None

        result = self.gait_extractor.extract_embedding(image_sequence)
        if result is None:
            return None
        
        raw_gait_embedding, view_type, gait_cycle_frames = result
        
        if raw_gait_embedding is None:
            return None
            
        gait_tensor = torch.from_numpy(raw_gait_embedding).float().to(self.device)
        
        # Reshape the tensor to be 2D (batch_size, features) for the linear layer.
        # The raw embedding is (C, H, W), so we flatten it.
        gait_tensor = gait_tensor.view(1, -1)

        with torch.no_grad():
            gait_embedding = self.gait_projection(gait_tensor).squeeze(0).cpu().numpy()

        # Normalize the final embedding
        norm = np.linalg.norm(gait_embedding)
        if norm > 0:
            gait_embedding /= norm
        
        return gait_embedding, view_type, gait_cycle_frames

class GaitAndShoeEmbedder:
    """
    A class to fuse gait and shoe embeddings into a single hybrid embedding.
    """

    def __init__(self):
        """
        Initializes the GaitAndShoeEmbedder.
        """
        # This class is a simple container for the fusion logic.
        # No model loading is needed here.
        pass

    def fuse_embeddings(self, gait_embedding, shoe_embedding):
        """
        Concatenates gait and shoe embeddings to create a fused representation.

        Args:
            gait_embedding (np.ndarray): The embedding for gait.
            shoe_embedding (np.ndarray): The embedding for the shoe.

        Returns:
            np.ndarray: The concatenated (fused) embedding.
                        Returns the gait embedding if the shoe embedding is not available.
                        Returns None if the gait embedding is not available.
        """
        if gait_embedding is None:
            return None
        
        if shoe_embedding is None:
            print("Shoe embedding not available. Returning gait-only embedding.")
            return gait_embedding
            
        # Ensure both embeddings are 1D arrays
        if gait_embedding.ndim > 1:
            gait_embedding = gait_embedding.flatten()
        if shoe_embedding.ndim > 1:
            shoe_embedding = shoe_embedding.flatten()

        # Concatenate the two embeddings
        fused_embedding = np.concatenate([gait_embedding, shoe_embedding])
        
        # Normalize the fused embedding to unit length
        embedding_norm = np.linalg.norm(fused_embedding)
        if embedding_norm > 0:
            fused_embedding = fused_embedding / embedding_norm
        
        print(f"Fused gait and shoe embeddings. Gait shape: {gait_embedding.shape}, Shoe shape: {shoe_embedding.shape}, Fused shape: {fused_embedding.shape}")

        return fused_embedding

if __name__ == '__main__':
    try:
        print("Creating Gait Embedder...")
        gait_embedder = GaitEmbedder()

        # --- Load Data ---
        # For this example, we'll use a single tracklet.
        # The first frame will be used for appearance, and the whole sequence for gait.
        track_id = "1"
        track_dir = Path(f"data/processed_output/tracked_persons/{track_id}")

        if not track_dir.exists() or not any(track_dir.iterdir()):
            print(f"Tracked images not found for ID {track_id}. Please run the tracking script first.")
        else:
            image_files = sorted(list(track_dir.glob("*.jpg"))) # Simpler sort
            
            # Load the image sequence
            image_sequence = [cv2.imread(str(f)) for f in image_files]
            
            if not image_sequence:
                print(f"No images found in {track_dir}")
            
            # If sequence is too short, duplicate frames for the example to run
            if len(image_sequence) > 0 and len(image_sequence) < 5:
                print(f"Warning: Only {len(image_sequence)} images found. Duplicating frames to meet minimum for gait analysis.")
                while len(image_sequence) < 5:
                    image_sequence.extend(image_sequence)
            
            image_sequence = image_sequence[:35] # Use up to 30 frames

            if len(image_sequence) >= 5:
                
                print("\nExtracting gait-based embedding...")
                result = gait_embedder.get_embedding(
                    image_sequence=image_sequence
                )
                
                if result:
                    final_embedding, view_type, gait_frames = result
                    print(f"\nSuccessfully created gait embedding for view: {view_type}.")
                    print(f"Detected gait cycle with {len(gait_frames)} frames.")
                    print(f"Final embedding shape: {final_embedding.shape}")
                    print(f"Final embedding L2 norm (should be ~1.0): {np.linalg.norm(final_embedding)}")
                else:
                    print("Failed to create gait embedding.")
            else:
                print("Not enough images in sequence for reliable gait extraction.")

    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
