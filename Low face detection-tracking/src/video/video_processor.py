"""
Video processing module for the Hybrid Person Tracking System.
Handles video input/output operations and frame processing.
"""

import cv2
import logging
import time
from typing import Tuple, Optional, Dict, Any

from ..config import config


class VideoProcessor:
    """
    Handles video capture and output operations.
    """
    
    def __init__(self, 
                 source: str,
                 output_path: str,
                 rtsp_buffer: int = config.RTSP_BUFFER,
                 display: bool = True):
        """
        Initialize video processor.
        
        Args:
            source: Path to video file or camera index (0 for webcam)
            output_path: Path to save output video
            rtsp_buffer: Buffer size for RTSP streams
            display: Whether to display video frames
        """
        self.source = source
        self.output_path = output_path
        self.rtsp_buffer = rtsp_buffer
        self.display = display
        self.cap = None
        self.out = None
        self.width = 0
        self.height = 0
        self.fps = 0
        
    def open(self) -> bool:
        """
        Open video source and initialize writer.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open video source
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                # Check if source is an RTSP URL
                is_rtsp = self.source.lower().startswith('rtsp://')
                
                # Open the video source
                self.cap = cv2.VideoCapture(self.source)
                
                # Set additional parameters for RTSP streams
                if is_rtsp:
                    # Set RTSP buffer size
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.rtsp_buffer)
                    
                    # Use TCP for RTSP transport (more reliable than UDP)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                    self.cap.set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TRANSPORT_TCP)
                    
                    logging.info(f"Connected to RTSP stream: {self.source}")
                
            if not self.cap.isOpened():
                raise IOError(f"Could not open video source: {self.source}")

            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            
            logging.info(f"Video source opened: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logging.error(f"Failed to open video source: {e}")
            return False
            
    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        Read a frame from the video source.
        
        Returns:
            Tuple[bool, Optional[cv2.Mat]]: Success flag and frame if successful
        """
        if self.cap is None:
            return False, None
            
        return self.cap.read()
        
    def write_frame(self, frame: cv2.Mat) -> None:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write
        """
        if self.out is not None:
            self.out.write(frame)
            
    def display_frame(self, frame: cv2.Mat, window_name: str = "Hybrid Person Tracking System") -> int:
        """
        Display a frame and handle key presses.
        
        Args:
            frame: Frame to display
            window_name: Name of display window
            
        Returns:
            int: Key code pressed or -1 if no key was pressed
        """
        if not self.display:
            return -1
            
        # Resize frame for display
        display_frame = self.resize_frame_for_display(frame)
        
        # Set window to be resizable
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        return cv2.waitKey(1) & 0xFF
            
    def resize_frame_for_display(self, frame: cv2.Mat) -> cv2.Mat:
        """
        Resize frame for display while maintaining aspect ratio.
        
        Args:
            frame: Original video frame
            
        Returns:
            Resized frame for display
        """
        # Get screen resolution
        screen_w = 1920  # Default screen width (fallback)
        screen_h = 1080  # Default screen height (fallback)
        
        try:
            # Try to get the actual screen resolution
            from screeninfo import get_monitors
            primary_monitor = get_monitors()[0]
            screen_w = primary_monitor.width
            screen_h = primary_monitor.height
        except:
            logging.warning("Could not get screen resolution, using default values")
        
        # Use 80% of screen size for maximum display area
        max_width = int(screen_w * 0.8)
        max_height = int(screen_h * 0.8)
        
        # Get current frame dimensions
        height, width = frame.shape[:2]
        
        # Calculate scaling factor to fit within screen bounds
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Only resize if necessary
        if scale < 1.0 or scale > 1.0:  # Allow upscaling too for small videos
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        
        return frame
        
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get video dimensions.
        
        Returns:
            Tuple[int, int]: Width and height
        """
        return self.width, self.height
        
    def get_fps(self) -> float:
        """
        Get video frame rate.
        
        Returns:
            float: Frames per second
        """
        return self.fps
        
    def release(self) -> None:
        """
        Release video capture and writer resources.
        """
        if self.cap is not None:
            self.cap.release()
            
        if self.out is not None:
            self.out.release()
            
        cv2.destroyAllWindows()
        logging.info("Video resources released") 