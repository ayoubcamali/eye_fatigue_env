"""
Face Detection Module using Simple Image Processing
No external ML models - just image processing algorithms
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Simplified face detection using contour analysis
    Works without external models in headless environments
    """
    
    def __init__(
        self,
        model_complexity: int = 1,
        static_image_mode: bool = False,
        max_num_faces: int = 5,
        refine_landmarks: bool = True
    ):
        """Initialize Face Detector"""
        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        logger.info("SimpleFaceDetector initialized successfully")
    
    def detect_faces(self, image: np.ndarray) -> Dict:
        """Detect faces using contour analysis"""
        if image is None or image.size == 0:
            return {"faces": [], "count": 0, "success": False}
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations to find face-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        faces = []
        for cnt in contours[:self.max_num_faces]:
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            # Filter by aspect ratio (faces are typically wider than tall)
            if w_cnt > 30 and h_cnt > 40 and 0.5 < w_cnt/h_cnt < 2:
                faces.append({
                    "bbox": (x, y, x + w_cnt, y + h_cnt),
                    "coordinates": (x, y, w_cnt, h_cnt)
                })
        
        return {
            "faces": faces,
            "count": len(faces),
            "success": True,
            "image_shape": image.shape
        }
    
    def detect_landmarks(self, image: np.ndarray) -> Dict:
        """Detect simplified landmarks in face regions"""
        if image is None or image.size == 0:
            return {"landmarks": [], "count": 0, "success": False}
        
        detections = self.detect_faces(image)
        landmarks_list = []
        
        for face_data in detections.get("faces", []):
            x1, y1, x2, y2 = face_data["bbox"]
            face_width = x2 - x1
            face_height = y2 - y1
            
            landmarks = []
            
            # Define approximate key points
            key_points = [
                ("left_eye", (0.35, 0.35)),
                ("right_eye", (0.65, 0.35)),
                ("nose", (0.5, 0.5)),
                ("left_mouth", (0.35, 0.75)),
                ("right_mouth", (0.65, 0.75)),
                ("chin", (0.5, 0.9)),
            ]
            
            for i, (name, (rel_x, rel_y)) in enumerate(key_points):
                x = int(x1 + rel_x * face_width)
                y = int(y1 + rel_y * face_height)
                landmarks.append({
                    "index": i,
                    "name": name,
                    "x": x,
                    "y": y,
                    "z": 0,
                    "normalized": {"x": rel_x, "y": rel_y, "z": 0}
                })
            
            # Add eye landmarks
            eye_landmarks = self._get_eye_landmarks(x1, y1, face_width, face_height)
            landmarks.extend(eye_landmarks)
            landmarks_list.append(landmarks)
        
        return {
            "landmarks": landmarks_list,
            "count": len(landmarks_list),
            "success": True,
            "image_shape": image.shape
        }
    
    def _get_eye_landmarks(self, x_offset, y_offset, face_width, face_height):
        """Generate eye region landmarks"""
        eye_landmarks = []
        
        for eye_idx, (rel_x_base) in enumerate([(0.35, "left"), (0.65, "right")]):
            rel_x, eye_name = rel_x_base
            
            eye_x = int(x_offset + rel_x * face_width)
            eye_y = int(y_offset + 0.35 * face_height)
            eye_w = int(0.2 * face_width)
            eye_h = int(0.15 * face_height)
            
            base_idx = 100 if eye_name == "left" else 200
            
            eye_landmarks.extend([
                {"index": base_idx, "name": f"{eye_name}_eye_left", "x": eye_x, "y": eye_y, "z": 0},
                {"index": base_idx+1, "name": f"{eye_name}_eye_right", "x": eye_x + eye_w, "y": eye_y, "z": 0},
                {"index": base_idx+2, "name": f"{eye_name}_eye_top", "x": eye_x + eye_w//2, "y": eye_y, "z": 0},
                {"index": base_idx+3, "name": f"{eye_name}_eye_bottom", "x": eye_x + eye_w//2, "y": eye_y + eye_h, "z": 0},
            ])
        
        return eye_landmarks
    
    def draw_detections(self, image, detections, draw_bbox=True, bbox_color=(0, 255, 0), thickness=2):
        """Draw face boxes"""
        output = image.copy()
        if draw_bbox:
            for face in detections.get("faces", []):
                x1, y1, x2, y2 = face.get("bbox")
                cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, thickness)
        return output
    
    def draw_landmarks(self, image, landmarks, draw_points=True, point_color=(0, 0, 255), point_radius=3):
        """Draw landmarks"""
        output = image.copy()
        if draw_points:
            for face_lms in landmarks.get("landmarks", []):
                for lm in face_lms:
                    x, y = int(lm["x"]), int(lm["y"])
                    color = (255, 0, 0) if "eye" in lm.get("name", "") else point_color
                    cv2.circle(output, (x, y), point_radius, color, -1)
        return output
    
    def get_face_regions(self, image, detections, padding=10):
        """Extract face crops"""
        regions = []
        for face in detections.get("faces", []):
            x1, y1, x2, y2 = face.get("bbox")
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(image.shape[1], int(x2) + padding)
            y2 = min(image.shape[0], int(y2) + padding)
            region = image[y1:y2, x1:x2]
            if region.size > 0:
                regions.append(region)
        return regions
    
    def release(self):
        """Release resources"""
        logger.info("FaceDetector resources released")


class WebcamFaceDetector:
    """Real-time face detection from webcam"""
    
    def __init__(self, camera_id=0):
        """Initialize"""
        self.detector = FaceDetector()
        self.camera_id = camera_id
        self.cap = None
        logger.info("WebcamFaceDetector initialized")
    
    def start(self, display=True):
        """Start detection loop"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        logger.info("Webcam started. Press 'q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                detections = self.detector.detect_faces(frame)
                landmarks = self.detector.detect_landmarks(frame)
                
                frame = self.detector.draw_detections(frame, detections)
                frame = self.detector.draw_landmarks(frame, landmarks)
                
                if display:
                    cv2.putText(frame, f"Faces: {detections['count']}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Face Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.stop()
    
    def stop(self):
        """Stop and cleanup"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        logger.info("Webcam stopped")
