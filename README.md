"""
README - Eye Fatigue Detection System
"""

# Eye Fatigue Detection System

A comprehensive Python application for real-time detection and monitoring of eye fatigue using computer vision and deep learning.

## 🎯 Features

- **Real-time Face Detection** using MediaPipe
- **Facial Landmark Detection** (468 landmarks per face)
- **Eye Fatigue Detection** based on:
  - Eye Aspect Ratio (EAR)
  - Blink Rate Detection
  - Eye Closure State
  - Pupil Position Tracking
- **Live Visualization** with:
  - Face bounding boxes
  - Facial landmarks
  - Fatigue level indicator
  - FPS counter
  - Blink counter
- **Video Recording** capability
- **Image Processing** support
- **Configurable Parameters** for different use cases

## 📋 Requirements

- Python 3.8+
- OpenCV (cv2)
- MediaPipe
- NumPy

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ayoubcamali/eye_fatigue_env.git
cd eye_fatigue_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Activate the virtual environment (if using):
```bash
source Lib/bin/activate  # On Linux/Mac
# or
Lib\Scripts\activate  # On Windows
```

## 💻 Usage

### Command Line

#### Real-time Webcam Detection
```bash
python main.py --camera 0
```

#### With Options
```bash
python main.py \
  --camera 0 \
  --save \
  --model-complexity 1 \
  --ear-threshold 0.2
```

#### Process Single Image
```bash
python main.py --image path/to/image.jpg --save
```

### Programmatic Usage

```python
from src.face_detector import FaceDetector
from src.eye_fatigue import EyeFatigueDetector
import cv2

# Initialize detectors
face_detector = FaceDetector()
fatigue_detector = EyeFatigueDetector()

# Load image
image = cv2.imread("image.jpg")

# Detect faces
detections = face_detector.detect_faces(image)
print(f"Faces detected: {detections['count']}")

# Detect landmarks
landmarks = face_detector.detect_landmarks(image)

# Detect fatigue
if landmarks['landmarks']:
    fatigue = fatigue_detector.detect_fatigue(landmarks['landmarks'][0])
    print(f"Fatigue Level: {fatigue['fatigue_level']:.2f}")
    print(f"Is Fatigued: {fatigue['is_fatigued']}")

# Draw results
output = face_detector.draw_detections(image, detections)
output = face_detector.draw_landmarks(output, landmarks)

cv2.imshow("Result", output)
cv2.waitKey(0)
```

## 📁 Project Structure

```
eye_fatigue_env/
├── main.py                 # Main application entry point
├── examples.py            # Example usage
├── requirements.txt       # Python dependencies
└── src/
    ├── __init__.py       # Package initialization
    ├── face_detector.py  # Face detection module
    ├── eye_fatigue.py    # Eye fatigue detection module
    ├── config.py         # Configuration settings
    └── utils.py          # Utility functions
```

## 🎮 Controls

During webcam detection:
- **'q'** - Quit application
- **'s'** - Save current frame
- **'r'** - Reset fatigue detection counter

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Camera settings
camera = CameraConfig(
    camera_id=0,
    frame_width=640,
    frame_height=480,
    fps=30
)

# Detection settings
detection = DetectionConfig(
    model_complexity=1,
    max_num_faces=1,
    refine_landmarks=True
)

# Fatigue thresholds
fatigue = FatigueConfig(
    ear_threshold=0.2,      # Eye Aspect Ratio threshold
    blink_threshold=3,      # Frames for blink detection
    fatigue_threshold=0.6   # Fatigue level threshold
)

# Display options
display = DisplayConfig(
    show_fps=True,
    show_landmarks=True,
    show_bbox=True,
    show_fatigue_info=True
)
```

## 📊 Output

The system generates:

1. **Real-time Display** with:
   - FPS counter
   - Detected faces count
   - Fatigue level (0-100%)
   - Blink counter
   - Eye state (open/closed)

2. **Output Files** (when `--save` is used):
   - Processed video: `output/eye_fatigue_TIMESTAMP.mp4`
   - Captured frames: `output/capture_TIMESTAMP.jpg`

3. **Logs**: `logs/` directory

## 🔧 API Reference

### FaceDetector

```python
detector = FaceDetector(
    model_complexity=1,
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True
)

# Detect faces
detections = detector.detect_faces(image)

# Detect landmarks
landmarks = detector.detect_landmarks(image)

# Draw results
output = detector.draw_detections(image, detections)
output = detector.draw_landmarks(output, landmarks)

# Extract face regions
faces = detector.get_face_regions(image, detections, padding=10)

# Release resources
detector.release()
```

### EyeFatigueDetector

```python
fatigue_detector = EyeFatigueDetector(
    ear_threshold=0.2,
    blink_threshold=3
)

# Detect fatigue
results = fatigue_detector.detect_fatigue(landmarks)

# Results structure
{
    "is_fatigued": bool,
    "fatigue_level": float,  # 0-1
    "indicators": {
        "blinks": int,
        "eye_aspect_ratio_left": float,
        "eye_aspect_ratio_right": float,
        "eye_closure_state": str  # "open" or "closed"
    }
}

# Reset counters
fatigue_detector.reset()
```

## 🎓 Understanding Eye Fatigue Detection

### Eye Aspect Ratio (EAR)

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

Where p1-p6 are the eye landmarks arranged as:
- p1, p4: Left and right eye corners
- p2, p3, p5, p6: Upper and lower eyelid points

EAR < 0.2 typically indicates closed or nearly closed eyes.

### Blink Detection

Tracks consecutive frames with EAR < threshold to identify blinks.

### Fatigue Indicators

- Low blink rate (< 0.2 blinks per frame)
- Prolonged eye closure
- Reduced Eye Aspect Ratio

## 🐛 Troubleshooting

### Camera not detected
```bash
python -c "import cv2; print(cv2.getBuildInformation())"
```

### Poor detection accuracy
- Increase `model_complexity` to 1
- Adjust lighting conditions
- Ensure face is clearly visible
- Try `--refine-landmarks` option

### High CPU usage
- Reduce `frame_width` and `frame_height`
- Lower `fps` setting
- Use lower `model_complexity`

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Ayoub Camali

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Last Updated**: 2026-08-31
