"""
Main application for Eye Fatigue Detection System
Real-time eye fatigue detection and monitoring from webcam
"""

import cv2
import logging
import argparse
from typing import Optional
import sys

from src.face_detector import FaceDetector, WebcamFaceDetector
from src.eye_fatigue import EyeFatigueDetector
from src.config import SystemConfig, CameraConfig, DetectionConfig, FatigueConfig, DisplayConfig
from src.utils import (
    draw_fps, create_fatigue_status_bar, create_info_panel,
    get_timestamp, create_output_directories, save_frame
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EyeFatigueApp:
    """
    Main application for eye fatigue detection
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the application
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self.face_detector = FaceDetector(
            model_complexity=self.config.detection.model_complexity,
            max_num_faces=self.config.detection.max_num_faces,
            refine_landmarks=self.config.detection.refine_landmarks
        )
        self.fatigue_detector = EyeFatigueDetector(
            ear_threshold=self.config.fatigue.ear_threshold,
            blink_threshold=self.config.fatigue.blink_threshold
        )
        
        # Create output directories
        create_output_directories(self.config)
        
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.is_running = False
        
        logger.info("EyeFatigueApp initialized")
    
    def process_frame(self, frame) -> tuple:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detection_results, fatigue_results)
        """
        output_frame = frame.copy()
        
        # Detect faces
        detections = self.face_detector.detect_faces(frame)
        
        # Extract detection results
        detection_info = {
            "faces_detected": detections.get("count", 0)
        }
        
        fatigue_info = {
            "is_fatigued": False,
            "fatigue_level": 0.0,
            "blinks": 0
        }
        
        # Process each detected face
        if detections.get("faces"):
            # Draw face detection boxes
            output_frame = self.face_detector.draw_detections(
                output_frame,
                detections,
                draw_bbox=self.config.display.show_bbox,
                bbox_color=self.config.display.bbox_color
            )
            
            # Detect landmarks for first face
            landmarks = self.face_detector.detect_landmarks(frame)
            
            if landmarks.get("landmarks"):
                # Draw landmarks
                output_frame = self.face_detector.draw_landmarks(
                    output_frame,
                    landmarks,
                    draw_points=self.config.display.show_landmarks,
                    point_color=self.config.display.landmark_color
                )
                
                # Detect fatigue from first face
                first_face_landmarks = landmarks["landmarks"][0]
                fatigue_results = self.fatigue_detector.detect_fatigue(first_face_landmarks)
                
                fatigue_info = {
                    "is_fatigued": fatigue_results.get("is_fatigued", False),
                    "fatigue_level": fatigue_results.get("fatigue_level", 0.0),
                    "blinks": fatigue_results.get("indicators", {}).get("blinks", 0),
                    "eye_closure": fatigue_results.get("indicators", {}).get("eye_closure_state", "unknown")
                }
        
        # Add visualization
        if self.config.display.show_fps:
            output_frame = draw_fps(output_frame, self.fps)
        
        if self.config.display.show_fatigue_info:
            output_frame = create_fatigue_status_bar(
                output_frame,
                fatigue_info["fatigue_level"],
                pos=(10, 70)
            )
            
            info_dict = {
                "Blinks": str(fatigue_info["blinks"]),
                "Eye State": fatigue_info["eye_closure"],
                "Status": "FATIGUED" if fatigue_info["is_fatigued"] else "NORMAL"
            }
            
            output_frame = create_info_panel(
                output_frame,
                info_dict,
                pos=(10, 130)
            )
        
        return output_frame, detection_info, fatigue_info
    
    def run_webcam(self, camera_id: int = 0, save_output: bool = False) -> None:
        """
        Run real-time detection from webcam
        
        Args:
            camera_id: Camera device ID
            save_output: Whether to save output video
        """
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        
        # Video writer if saving
        writer = None
        if save_output:
            from src.utils import get_video_writer
            output_path = f"{self.config.output_dir}/eye_fatigue_{get_timestamp().replace(':', '-')}.mp4"
            writer = get_video_writer(
                output_path,
                fps=self.config.camera.fps,
                frame_size=(
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                )
            )
        
        self.is_running = True
        
        logger.info(f"Starting webcam detection (camera_id={camera_id})")
        logger.info("Press 'q' to quit, 's' to save frame, 'r' to reset")
        
        import time
        prev_time = time.time()
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                output_frame, detection_info, fatigue_info = self.process_frame(frame)
                
                # Calculate FPS
                curr_time = time.time()
                self.fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                self.frame_count += 1
                
                # Display frame
                cv2.imshow(self.config.display.window_name, output_frame)
                
                # Save to video if recording
                if writer:
                    writer.write(output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('s'):  # Save frame
                    save_frame(output_frame, self.config.output_dir, "capture")
                elif key == ord('r'):  # Reset
                    self.fatigue_detector.reset()
                    logger.info("Fatigue detection reset")
        
        finally:
            self.stop(writer)
    
    def run_image(self, image_path: str, save_output: bool = False) -> None:
        """
        Process a single image
        
        Args:
            image_path: Path to image file
            save_output: Whether to save output image
        """
        logger.info(f"Processing image: {image_path}")
        
        frame = cv2.imread(image_path)
        
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return
        
        # Process frame
        output_frame, detection_info, fatigue_info = self.process_frame(frame)
        
        # Display
        cv2.imshow(self.config.display.window_name, output_frame)
        
        logger.info(f"Detection Info: {detection_info}")
        logger.info(f"Fatigue Info: {fatigue_info}")
        
        if save_output:
            save_frame(output_frame, self.config.output_dir, "result")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def stop(self, writer=None) -> None:
        """
        Stop the application and release resources
        
        Args:
            writer: Video writer object if recording
        """
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        if writer:
            writer.release()
            logger.info("Video saved")
        
        cv2.destroyAllWindows()
        self.face_detector.release()
        self.fatigue_detector.reset()
        
        logger.info(f"Application stopped. Processed {self.frame_count} frames")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Eye Fatigue Detection System"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for processing"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video/image"
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        choices=[0, 1],
        default=1,
        help="Model complexity (0 or 1)"
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.2,
        help="Eye Aspect Ratio threshold"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig(
        detection=DetectionConfig(model_complexity=args.model_complexity),
        fatigue=FatigueConfig(ear_threshold=args.ear_threshold)
    )
    
    # Create and run application
    app = EyeFatigueApp(config)
    
    if args.image:
        app.run_image(args.image, save_output=args.save)
    else:
        app.run_webcam(camera_id=args.camera, save_output=args.save)


if __name__ == "__main__":
    main()
