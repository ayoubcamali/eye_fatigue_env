"""
Eye Fatigue Detection Module
Detects various indicators of eye fatigue from facial landmarks
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EyeFatigueDetector:
    """
    Detects eye fatigue indicators including:
    - Eye aspect ratio (EAR)
    - Blink rate
    - Eye pupil position
    - Eyelid closure
    """
    
    # Eye landmarks indices for left and right eye
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 155, 154]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 362, 386, 385]
    
    # Eye region landmarks for better detection
    LEFT_EYE_FULL = list(range(33, 133))
    RIGHT_EYE_FULL = list(range(133, 200))
    
    # Pupil landmarks
    LEFT_PUPIL = 468
    RIGHT_PUPIL = 473
    
    def __init__(self, ear_threshold: float = 0.2, blink_threshold: int = 3):
        """
        Initialize Eye Fatigue Detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold for closed eyes
            blink_threshold: Consecutive frames for blink detection
        """
        self.ear_threshold = ear_threshold
        self.blink_threshold = blink_threshold
        self.consecutive_frames = 0
        self.blink_count = 0
        self.last_state = "open"
        self.frame_count = 0
        
        logger.info("EyeFatigueDetector initialized")
    
    @staticmethod
    def calculate_eye_aspect_ratio(
        eye_points: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        Formula: ((p2-p6) + (p3-p5)) / (2 * (p1-p4))
        
        Args:
            eye_points: 6 points of eye contour (left to right, top to bottom)
            
        Returns:
            Eye Aspect Ratio value
        """
        if len(eye_points) != 6:
            return 0.0
        
        # Calculate vertical distances
        vertical_dist_1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        vertical_dist_2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Calculate horizontal distance
        horizontal_dist = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        # Calculate aspect ratio
        ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
        
        return ear
    
    def extract_eye_points(
        self,
        landmarks: List[Dict]
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Extract eye points from facial landmarks
        
        Args:
            landmarks: List of facial landmark dictionaries
            
        Returns:
            Tuple of (left_eye_points, right_eye_points)
        """
        left_eye = []
        right_eye = []
        
        try:
            # Extract left eye points
            for idx in [33, 160, 158, 133, 155, 154]:
                if idx < len(landmarks):
                    pt = (landmarks[idx]["x"], landmarks[idx]["y"])
                    left_eye.append(pt)
            
            # Extract right eye points
            for idx in [362, 385, 387, 362, 386, 385]:
                if idx < len(landmarks):
                    pt = (landmarks[idx]["x"], landmarks[idx]["y"])
                    right_eye.append(pt)
        
        except (IndexError, KeyError) as e:
            logger.error(f"Error extracting eye points: {e}")
            return ([], [])
        
        return (left_eye, right_eye)
    
    def detect_fatigue(self, landmarks: List[Dict]) -> Dict:
        """
        Detect eye fatigue indicators
        
        Args:
            landmarks: Facial landmarks from face detection
            
        Returns:
            Dictionary with fatigue detection results
        """
        results = {
            "is_fatigued": False,
            "fatigue_level": 0.0,
            "indicators": {
                "blinks": self.blink_count,
                "eye_aspect_ratio_left": 0.0,
                "eye_aspect_ratio_right": 0.0,
                "eye_closure_state": "open"
            }
        }
        
        if not landmarks:
            return results
        
        try:
            left_eye_pts, right_eye_pts = self.extract_eye_points(landmarks)
            
            if not left_eye_pts or not right_eye_pts:
                return results
            
            # Calculate Eye Aspect Ratio
            ear_left = self.calculate_eye_aspect_ratio(left_eye_pts)
            ear_right = self.calculate_eye_aspect_ratio(right_eye_pts)
            
            results["indicators"]["eye_aspect_ratio_left"] = ear_left
            results["indicators"]["eye_aspect_ratio_right"] = ear_right
            
            avg_ear = (ear_left + ear_right) / 2.0
            
            # Detect blinks
            if avg_ear < self.ear_threshold:
                self.consecutive_frames += 1
                results["indicators"]["eye_closure_state"] = "closed"
                
                if self.last_state == "open" and self.consecutive_frames >= self.blink_threshold:
                    self.blink_count += 1
                    results["indicators"]["blinks"] = self.blink_count
            else:
                self.consecutive_frames = 0
                self.last_state = "open"
                results["indicators"]["eye_closure_state"] = "open"
            
            # Calculate fatigue level (0-1)
            # Lower EAR indicates more closed eyes
            fatigue_level = 1.0 - min(avg_ear / 0.4, 1.0)
            
            # Apply blink count factor
            # Too few blinks can indicate fatigue
            blink_rate = self.blink_count / max(self.frame_count, 1)
            if blink_rate < 0.2:  # Less than 0.2 blinks per frame
                fatigue_level = min(fatigue_level + 0.3, 1.0)
            
            results["fatigue_level"] = fatigue_level
            results["is_fatigued"] = fatigue_level > 0.6
            
            self.frame_count += 1
        
        except Exception as e:
            logger.error(f"Error detecting fatigue: {e}")
        
        return results
    
    def reset(self):
        """Reset fatigue detection counters"""
        self.consecutive_frames = 0
        self.blink_count = 0
        self.frame_count = 0
        self.last_state = "open"
        logger.info("Fatigue detection reset")
