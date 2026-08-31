"""
Utility functions for the eye fatigue detection system
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import os

logger = logging.getLogger(__name__)


def draw_text_with_bg(
    image: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.7,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    padding: int = 5
) -> np.ndarray:
    """
    Draw text with background rectangle
    
    Args:
        image: Input image
        text: Text to draw
        pos: Text position (x, y)
        font: Font type
        font_scale: Font scale
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        thickness: Text thickness
        padding: Background padding
        
    Returns:
        Image with drawn text
    """
    x, y = pos
    
    # Get text size
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    
    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )
    
    return image


def draw_fps(
    image: np.ndarray,
    fps: float,
    pos: Tuple[int, int] = (10, 30),
    font_scale: float = 0.7
) -> np.ndarray:
    """
    Draw FPS on image
    
    Args:
        image: Input image
        fps: Frames per second
        pos: Text position
        font_scale: Font scale
        
    Returns:
        Image with FPS drawn
    """
    fps_text = f"FPS: {fps:.1f}"
    return draw_text_with_bg(
        image,
        fps_text,
        pos,
        font_scale=font_scale,
        text_color=(0, 255, 0)
    )


def create_fatigue_status_bar(
    image: np.ndarray,
    fatigue_level: float,
    bar_width: int = 200,
    bar_height: int = 30,
    pos: Tuple[int, int] = (10, 70)
) -> np.ndarray:
    """
    Create a fatigue level status bar
    
    Args:
        image: Input image
        fatigue_level: Fatigue level (0-1)
        bar_width: Width of status bar
        bar_height: Height of status bar
        pos: Position of status bar
        
    Returns:
        Image with status bar
    """
    x, y = pos
    
    # Draw background
    cv2.rectangle(
        image,
        (x, y),
        (x + bar_width, y + bar_height),
        (50, 50, 50),
        -1
    )
    
    # Draw border
    cv2.rectangle(
        image,
        (x, y),
        (x + bar_width, y + bar_height),
        (200, 200, 200),
        2
    )
    
    # Draw filled bar
    fill_width = int(bar_width * min(fatigue_level, 1.0))
    
    if fatigue_level < 0.5:
        color = (0, 255, 0)  # Green
    elif fatigue_level < 0.8:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red
    
    cv2.rectangle(
        image,
        (x, y),
        (x + fill_width, y + bar_height),
        color,
        -1
    )
    
    # Draw percentage text
    percentage = int(fatigue_level * 100)
    text = f"{percentage}%"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    text_x = x + (bar_width - text_size[0]) // 2
    text_y = y + (bar_height + text_size[1]) // 2
    
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return image


def create_info_panel(
    image: np.ndarray,
    info_dict: Dict[str, str],
    pos: Tuple[int, int] = (10, 150),
    line_height: int = 30,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Create an information panel with multiple lines
    
    Args:
        image: Input image
        info_dict: Dictionary of information to display
        pos: Starting position
        line_height: Height between lines
        font_scale: Font scale
        
    Returns:
        Image with info panel
    """
    x, y = pos
    
    for i, (key, value) in enumerate(info_dict.items()):
        text = f"{key}: {value}"
        draw_text_with_bg(
            image,
            text,
            (x, y + i * line_height),
            font_scale=font_scale
        )
    
    return image


def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_output_directories(config) -> None:
    """
    Create necessary output directories
    
    Args:
        config: System configuration
    """
    directories = [config.output_dir, config.log_dir]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def save_frame(
    frame: np.ndarray,
    output_dir: str = "output",
    prefix: str = "frame"
) -> Optional[str]:
    """
    Save a frame with timestamp
    
    Args:
        frame: Frame to save
        output_dir: Output directory
        prefix: File prefix
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        cv2.imwrite(filepath, frame)
        logger.info(f"Frame saved: {filepath}")
        
        return filepath
    
    except Exception as e:
        logger.error(f"Failed to save frame: {e}")
        return None


def get_video_writer(
    output_path: str,
    fourcc_code: str = "mp4v",
    fps: int = 30,
    frame_size: Tuple[int, int] = (640, 480)
) -> Optional[cv2.VideoWriter]:
    """
    Create a video writer object
    
    Args:
        output_path: Path for output video
        fourcc_code: FourCC codec code
        fps: Frames per second
        frame_size: Frame size (width, height)
        
    Returns:
        cv2.VideoWriter object or None
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not writer.isOpened():
            logger.error("Failed to create video writer")
            return None
        
        logger.info(f"Video writer created: {output_path}")
        return writer
    
    except Exception as e:
        logger.error(f"Error creating video writer: {e}")
        return None


def convert_color_space(
    image: np.ndarray,
    from_color: str = "BGR",
    to_color: str = "RGB"
) -> np.ndarray:
    """
    Convert between color spaces
    
    Args:
        image: Input image
        from_color: Source color space
        to_color: Target color space
        
    Returns:
        Converted image
    """
    if from_color == "BGR" and to_color == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif from_color == "RGB" and to_color == "BGR":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif from_color == "BGR" and to_color == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif from_color == "RGB" and to_color == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    return image


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        width: Target width
        height: Target height
        scale: Scale factor (0-1)
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if scale:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width and height:
        new_w, new_h = width, height
    elif width:
        new_w = width
        new_h = int(h * (width / w))
    elif height:
        new_h = height
        new_w = int(w * (height / h))
    else:
        return frame
    
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
