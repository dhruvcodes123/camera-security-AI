import cv2
import torch
from ultralytics import YOLO

class ShoeDetector:
    """
    A class to detect shoes in an image using a YOLOv8 model.
    """

    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.25):
        """
        Initializes the ShoeDetector.

        Args:
            model_path (str): The path to the YOLOv8 model file.
            confidence_threshold (float): The confidence threshold for shoe detections.
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # print(f"Using device for shoe detection: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.confidence_threshold = confidence_threshold

    def detect_shoes(self, frame):
        """
        Detects shoes in a given frame.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            list: A list of tuples, where each tuple contains (bounding_box, confidence).
                  Returns an empty list if no shoes are detected.
        """
        # We assume the shoe class is implicitly handled if the model is trained for it.
        # If the model is a general one, you might need to filter by a class ID.
        # For this implementation, we will treat all detections as shoes.
        results = self.model(frame, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    detections.append(((x1, y1, x2, y2), confidence))
        
        return detections

    def get_best_shoe_crop(self, frame):
        """
        Detects shoes in a frame and returns the crop of the best detection.
        The best detection is determined by the highest confidence score and largest area.

        Args:
            frame (np.ndarray): The input image frame.

        Returns:
            np.ndarray: The cropped image of the best shoe detection.
                        Returns None if no shoes are detected.
        """
        detections = self.detect_shoes(frame)
        if not detections:
            return None

        # Sort by confidence and then by bounding box area (width * height)
        best_detection = sorted(
            detections,
            key=lambda d: (d[1], (d[0][2] - d[0][0]) * (d[0][3] - d[0][1])),
            reverse=True
        )[0]
        
        x1, y1, x2, y2 = best_detection[0]
        
        # Ensure the crop coordinates are valid
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            # print(f"Warning: Invalid crop coordinates {best_detection[0]} for frame shape {frame.shape}")
            return None
            
        crop = frame[y1:y2, x1:x2]
        
        return crop 