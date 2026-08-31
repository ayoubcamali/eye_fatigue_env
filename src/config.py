"""
Configuration settings for eye fatigue detection system
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraConfig:
    """Camera configuration"""
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30


@dataclass
class DetectionConfig:
    """Face and eye detection configuration"""
    model_complexity: int = 1
    max_num_faces: int = 1
    refine_landmarks: bool = True
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5


@dataclass
class FatigueConfig:
    """Eye fatigue detection configuration"""
    ear_threshold: float = 0.2  # Eye Aspect Ratio threshold
    blink_threshold: int = 3    # Consecutive frames for blink
    fatigue_threshold: float = 0.6  # Fatigue level threshold
    min_blink_rate: float = 0.2  # Minimum blinks per frame


@dataclass
class DisplayConfig:
    """Display configuration"""
    show_fps: bool = True
    show_landmarks: bool = True
    show_bbox: bool = True
    show_fatigue_info: bool = True
    window_name: str = "Eye Fatigue Detection"
    
    # Colors (BGR format)
    bbox_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    landmark_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    fatigue_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    normal_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    
    # Font
    font: int = 0  # cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.7
    font_thickness: int = 2


@dataclass
class SystemConfig:
    """System configuration"""
    log_level: str = "INFO"
    save_logs: bool = True
    log_dir: str = "logs"
    output_dir: str = "output"
    
    # Sub-configurations
    camera: CameraConfig = None
    detection: DetectionConfig = None
    fatigue: FatigueConfig = None
    display: DisplayConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations"""
        if self.camera is None:
            self.camera = CameraConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.fatigue is None:
            self.fatigue = FatigueConfig()
        if self.display is None:
            self.display = DisplayConfig()


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()
