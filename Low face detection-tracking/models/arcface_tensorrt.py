import cv2
import numpy as np
import logging
from typing import Tuple, Optional

from .tensorrt_model import TensorRTModel
from src.utils.helpers import face_alignment

__all__ = ["ArcFace_TensorRT"]


class ArcFace_TensorRT:
    """
    TensorRT-optimized ArcFace Model for Face Recognition on Jetson
    
    This class implements a face encoder using the ArcFace architecture,
    optimized with TensorRT for faster inference on Jetson devices.
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ArcFace TensorRT face encoder model.

        Args:
            model_path (str): Path to ONNX model file (will be converted to TensorRT if needed).

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.logger = logging.getLogger("ArcFace_TensorRT")
        self.model_path = model_path
        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_scale = 127.5
        self.embedding_size = 512  # Default for ArcFace

        self.logger.info(f"Initializing ArcFace TensorRT model from {self.model_path}")

        # Check for TensorRT engine file
        engine_path = model_path.replace('.onnx', '.engine')
        
        try:
            # Initialize TensorRT model
            self.model = TensorRTModel(engine_path=engine_path, onnx_path=model_path)
            
            # Set input and output names
            self.input_name = "data"  # Default input name for ArcFace
            self.output_names = ["fc1"]  # Default output name for ArcFace
            
            self.logger.info(f"Successfully initialized TensorRT face encoder from {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TensorRT face encoder model from '{self.model_path}'", exc_info=True)
            raise RuntimeError(f"Failed to initialize TensorRT model for '{self.model_path}'") from e

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess the face image: resize, normalize, and convert to the required format.

        Args:
            face_image (np.ndarray): Input face image in BGR format.

        Returns:
            np.ndarray: Preprocessed image blob ready for inference.
        """
        resized_face = cv2.resize(face_image, self.input_size)
        
        # Single-value normalization using cv2.dnn
        face_blob = cv2.dnn.blobFromImage(
            resized_face,
            scalefactor=1.0 / self.normalization_scale,
            size=self.input_size,
            mean=(self.normalization_mean,)*3,
            swapRB=True
        )
        
        return face_blob

    def get_embedding(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None, 
                      normalized: bool = True) -> np.ndarray:
        """
        Extract face embedding from the given face image.

        Args:
            face_image (np.ndarray): Input face image.
            landmarks (Optional[np.ndarray]): Optional 5-point facial landmarks for alignment.
            normalized (bool): Whether to normalize the output embedding.

        Returns:
            np.ndarray: Face embedding vector.
        """
        # Apply face alignment if landmarks are provided
        if landmarks is not None and landmarks.size >= 10:
            aligned_face = face_alignment(face_image, landmarks, self.input_size)
            if aligned_face is None:
                self.logger.warning("Face alignment failed, using original image")
                aligned_face = face_image
        else:
            aligned_face = face_image

        # Preprocess the face image
        face_blob = self.preprocess(aligned_face)
        
        # Run inference using TensorRT
        outputs = self.model.infer(face_blob)
        embedding = outputs[0].flatten()
        
        # Normalize embedding if requested
        if normalized and np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
            
        return embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): First face embedding.
            embedding2 (np.ndarray): Second face embedding.

        Returns:
            float: Cosine similarity score (higher means more similar).
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
            
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(cosine_sim) 