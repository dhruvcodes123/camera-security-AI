import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import yaml
from tqdm import tqdm

class MobileDetectionInference:
    def __init__(self, model_path, device='auto', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the inference class
        
        Args:
            model_path (str): Path to the trained model weights
            device (str): Device to run inference on ('auto', 'cuda', 'cpu')
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        try:
            # Load the model weights
            self.model = torch.load(self.model_path, map_location=self.device)
            
            # If it's a state dict, we need to create the model architecture first
            if isinstance(self.model, dict):
                # This is likely a state dict, we need to create the model first
                # For now, we'll assume it's a complete model
                print("Loaded model state dict")
            else:
                print("Loaded complete model")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Enable CUDA optimization if available
            if self.device.type == 'cuda':
                self.model = torch.jit.optimize_for_inference(self.model) if hasattr(torch.jit, 'optimize_for_inference') else self.model
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize to model input size (assuming 640x640 for YOLO models)
        input_size = (640, 640)
        image_resized = image.resize(input_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(image_resized)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW and normalize
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_predictions(self, predictions, original_shape):
        """
        Postprocess model predictions
        
        Args:
            predictions: Raw model predictions
            original_shape: Original image shape (height, width)
            
        Returns:
            list: List of detections with bounding boxes, confidence, and class
        """
        # This is a simplified postprocessing - you may need to adjust based on your model
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0] if len(predictions) > 0 else predictions
        
        # Apply confidence threshold
        conf_mask = predictions[..., 4] > self.conf_threshold
        predictions = predictions[conf_mask]
        
        if len(predictions) == 0:
            return []
        
        # Convert to CPU for postprocessing
        predictions = predictions.cpu().numpy()
        
        # Extract bounding boxes, confidence, and class predictions
        boxes = predictions[:, :4]
        confidences = predictions[:, 4]
        class_ids = predictions[:, 5] if predictions.shape[1] > 5 else np.zeros(len(predictions))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detection = {
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': int(class_ids[i])
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, image, detections, class_names=None):
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detections
            class_names: List of class names
            
        Returns:
            numpy.ndarray: Image with detections drawn
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image_draw = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Class {class_id}: {confidence:.2f}"
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {confidence:.2f}"
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(image_draw, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(image_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image_draw
    
    def predict_image(self, image_path, output_path=None, show_result=True):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image
            show_result (bool): Whether to display the result
            
        Returns:
            list: List of detections
        """
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        original_shape = image.size[::-1]  # (height, width)
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            start_time = time.time()
            predictions = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time:.3f} seconds")
        
        # Postprocess
        detections = self.postprocess_predictions(predictions, original_shape)
        
        # Draw detections
        result_image = self.draw_detections(image, detections)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"Result saved to: {output_path}")
        
        # Show result
        if show_result:
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.title(f"Detections: {len(detections)}")
            plt.axis('off')
            plt.show()
        
        return detections
    
    def predict_video(self, video_path, output_path=None, show_result=True):
        """
        Run inference on a video
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video
            show_result (bool): Whether to display the result
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_inference_time = 0
        
        # Process frames
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                # Run inference
                start_time = time.time()
                detections = self.predict_image(image, show_result=False)
                inference_time = time.time() - start_time
                
                total_inference_time += inference_time
                frame_count += 1
                
                # Draw detections
                result_frame = self.draw_detections(image, detections)
                
                # Convert back to BGR for OpenCV
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                if output_path:
                    out.write(result_frame_bgr)
                
                # Show frame
                if show_result:
                    cv2.imshow('Detection Result', result_frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
                pbar.set_postfix({'FPS': f'{1/inference_time:.1f}'})
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_inference_time = total_inference_time / frame_count
        print(f"Average inference time: {avg_inference_time:.3f} seconds")
        print(f"Average FPS: {1/avg_inference_time:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Mobile Detection Inference')
    parser.add_argument('--model', type=str, default='model/weights_mobile_using_detection.pt',
                       help='Path to model weights')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or video')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display results')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = MobileDetectionInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    # Check if input is image or video
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Determine file type
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    is_video = input_path.suffix.lower() in video_extensions
    
    if is_video:
        print("Processing video...")
        inference.predict_video(
            video_path=str(input_path),
            output_path=args.output,
            show_result=not args.no_display
        )
    else:
        print("Processing image...")
        inference.predict_image(
            image_path=str(input_path),
            output_path=args.output,
            show_result=not args.no_display
        )

if __name__ == "__main__":
    main() 