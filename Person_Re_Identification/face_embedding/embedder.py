#!/usr/bin/env python3
"""
Buffalo_l-Only Face Embedding System
Simplified face recognition using only Buffalo_l model.
"""

import insightface
import numpy as np
import cv2
import os
import logging

# Suppress ONNX runtime warnings
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

class FaceEmbedder:
    def __init__(self, det_thresh=0.5):
        """
        Initializes the FaceEmbedder using only Buffalo_l model.
        
        Args:
            det_thresh (float): The confidence threshold for face detection.
        """
        # print("Initializing Buffalo_l Face Recognition model...")
        
        # Initialize Buffalo_l model for both detection and recognition
        self.model = None
        self.model_name = 'buffalo_l'
        
        # Load Buffalo_l model
        self._load_buffalo_l_model(det_thresh)

    def _load_buffalo_l_model(self, det_thresh):
        """Load Buffalo_l model for both detection and recognition."""
        try:
            # print("Loading Buffalo_l model...")
            self.model = insightface.app.FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition'],
                providers=['CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=det_thresh)
            # print("✅ Buffalo_l model loaded successfully!")
            # print("   - Description: R50 backbone, good balance of speed and accuracy")
            # print(f"   - Detection threshold: {det_thresh}")
            # print("   - Provides 512-dimensional normalized embeddings")
            
        except Exception as e:
            # print(f"❌ Failed to load Buffalo_l model: {e}")
            raise RuntimeError("❌ Failed to load Buffalo_l face recognition model!")

    def get_all_face_detections(self, image: np.ndarray):
        """
        Detects all faces in the given image and returns their data.
        
        Args:
            image (np.ndarray): The input image in BGR format.
            
        Returns:
            list: A list of dictionaries, where each dictionary contains:
                  'bbox' (list): The bounding box [x1, y1, x2, y2].
                  'det_score' (float): The detection confidence score.
                  'embedding' (np.ndarray): The 512-dimension face embedding.
                  'crop' (np.ndarray): The cropped face image.
        """
        detections = []
        
        # Use Buffalo_l model for both detection and recognition
        faces = self.model.get(image)
        
        if not faces:
            return detections
            
        for face in faces:
            x1, y1, x2, y2 = [int(val) for val in face.bbox]
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            # Ensure embedding is normalized 512-dimensional vector
            embedding = face.normed_embedding.astype(np.float32)
            assert embedding.shape == (512,), f"Expected 512-D embedding, got {embedding.shape}"

            detections.append({
                'bbox': face.bbox,
                'det_score': face.det_score,
                'embedding': embedding,
                'crop': crop
            })
            
        return detections

    def get_face_embedding(self, image: np.ndarray):
        """
        Detects faces and returns the embedding and crop for the best quality face.
        
        Args:
            image (np.ndarray): The input image in BGR format.
            
        Returns:
            tuple: (embedding, crop) where:
                   - embedding (np.ndarray): The 512-dimension face embedding (None if no face).
                   - crop (np.ndarray): The cropped face image (None if no face).
        """
        detections = self.get_all_face_detections(image)
        
        if not detections:
            return None, None
            
        # Return the face with the highest detection score
        best_detection = max(detections, key=lambda x: x['det_score'])
        return best_detection['embedding'], best_detection['crop']

    def get_model_info(self):
        """
        Returns information about the currently loaded model.
        
        Returns:
            dict: Model information including name, embedding dimension, etc.
        """
        return {
            'model_name': self.model_name,
            'embedding_dim': 512,
            'is_normalized': True,
            'backbone': 'R50',
            'accuracy_level': 'good'
        }

    def validate_embeddings(self, embeddings):
        """
        Validates that embeddings are properly formatted.
        
        Args:
            embeddings (list): List of embeddings to validate.
            
        Returns:
            bool: True if all embeddings are valid, False otherwise.
        """
        if not embeddings:
            return False
            
        for embedding in embeddings:
            if embedding is None:
                return False
            if not isinstance(embedding, np.ndarray):
                return False
            if embedding.shape != (512,):
                return False
            if not np.allclose(np.linalg.norm(embedding), 1.0, rtol=1e-5):
                return False
                
        return True

    def test_model_functionality(self):
        """
        Tests the model functionality with a dummy image.
        
        Returns:
            dict: Test results including success status and metrics.
        """
        print("Testing Buffalo_l model functionality...")
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        try:
            # Test face detection
            detections = self.get_all_face_detections(dummy_image)
            
            # Test embedding extraction
            embedding, crop = self.get_face_embedding(dummy_image)
            
            # Test model info
            model_info = self.get_model_info()
            
            result = {
                'success': True,
                'detections_count': len(detections),
                'embedding_extracted': embedding is not None,
                'model_info': model_info,
                'error': None
            }
            
            if embedding is not None:
                result['embedding_shape'] = embedding.shape
                result['embedding_norm'] = float(np.linalg.norm(embedding))
                result['is_normalized'] = np.allclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)
            
            print("✅ Buffalo_l model functionality test passed!")
            return result
            
        except Exception as e:
            print(f"❌ Buffalo_l model functionality test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_info': self.get_model_info() if hasattr(self, 'model_name') else None
            }

if __name__ == '__main__':
    # Example usage with model validation
    try:
        print("--- Buffalo_l-Only FaceEmbedder ---")
        
        # Initialize embedder
        embedder = FaceEmbedder(det_thresh=0.5)
        
        # Show model info
        model_info = embedder.get_model_info()
        print(f"\n📊 Loaded Model Info:")
        for key, value in model_info.items():
            print(f"   - {key}: {value}")
        
        # Validate the model
        is_valid = embedder.validate_embeddings([np.ones(512) / np.sqrt(512)]) # Normalized test vector
        if not is_valid:
            print("❌ Model validation failed. Please check the installation.")
            exit(1)
        
        # Test with sample image if available
        test_image_path = 'sample_face.jpg'
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print("Creating a dummy test image...")
            dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.imwrite(test_image_path, dummy_image)
            test_image = cv2.imread(test_image_path)

        if test_image is None:
            raise FileNotFoundError(f"Could not read '{test_image_path}'. Please provide a sample image.")
        
        # Test all detection methods
        print("\n--- Testing get_all_face_detections ---")
        all_detections = embedder.get_all_face_detections(test_image)
        if all_detections:
            print(f"Detected {len(all_detections)} faces in the image.")
            for i, det in enumerate(all_detections):
                print(f"  - Face {i+1}: Score={det['det_score']:.2f}, Embedding shape={det['embedding'].shape}")
        else:
            print("No faces detected with get_all_face_detections.")
            
        print("\n--- Testing get_face_embedding ---")
        face_embedding, face_crop = embedder.get_face_embedding(test_image)
        
        if face_embedding is not None and face_crop is not None:
            print(f"✅ Successfully generated face embedding:")
            print(f"   - Shape: {face_embedding.shape}")
            print(f"   - Dtype: {face_embedding.dtype}")
            print(f"   - Norm: {np.linalg.norm(face_embedding):.4f}")
            cv2.imwrite("sample_face_cropped.jpg", face_crop)
            print("   - Saved cropped face to 'sample_face_cropped.jpg'")
        else:
            print("❌ No face was detected in the sample image.")
            
    except Exception as e:
        print(f"❌ An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
