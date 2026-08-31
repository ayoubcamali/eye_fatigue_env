"""
Example usage of the Eye Fatigue Detection system
"""

import cv2
from src.face_detector import FaceDetector, WebcamFaceDetector
from src.eye_fatigue import EyeFatigueDetector


def example_face_detection():
    """Example: Basic face detection"""
    print("=== Face Detection Example ===")
    
    # Initialize detector
    detector = FaceDetector(model_complexity=1)
    
    # Load image
    image = cv2.imread("sample_image.jpg")  # Replace with actual image
    
    if image is None:
        print("Image not found. Using webcam instead...")
        return
    
    # Detect faces
    detections = detector.detect_faces(image)
    print(f"Faces detected: {detections['count']}")
    
    # Draw detections
    output = detector.draw_detections(image, detections)
    
    # Display
    cv2.imshow("Face Detection", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    detector.release()


def example_landmark_detection():
    """Example: Facial landmark detection"""
    print("=== Landmark Detection Example ===")
    
    detector = FaceDetector()
    
    # Get from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not available")
        return
    
    try:
        for i in range(10):  # 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect landmarks
            landmarks = detector.detect_landmarks(frame)
            print(f"Frame {i}: {landmarks['count']} faces with landmarks")
            
            # Draw landmarks
            output = detector.draw_landmarks(frame, landmarks)
            
            cv2.imshow("Landmarks", output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()


def example_eye_fatigue():
    """Example: Eye fatigue detection"""
    print("=== Eye Fatigue Detection Example ===")
    
    face_detector = FaceDetector()
    fatigue_detector = EyeFatigueDetector()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not available")
        return
    
    try:
        for i in range(100):  # 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect landmarks
            landmarks = face_detector.detect_landmarks(frame)
            
            if landmarks.get("landmarks"):
                # Get first face
                face_landmarks = landmarks["landmarks"][0]
                
                # Detect fatigue
                fatigue_results = fatigue_detector.detect_fatigue(face_landmarks)
                
                print(f"Frame {i}: Fatigue Level = {fatigue_results['fatigue_level']:.2f}")
                print(f"  - Blinks: {fatigue_results['indicators']['blinks']}")
                print(f"  - EAR Left: {fatigue_results['indicators']['eye_aspect_ratio_left']:.3f}")
                print(f"  - EAR Right: {fatigue_results['indicators']['eye_aspect_ratio_right']:.3f}")
                
                if fatigue_results['is_fatigued']:
                    print("  ⚠️  FATIGUE DETECTED!")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.release()
        fatigue_detector.reset()


def example_webcam_detection():
    """Example: Real-time webcam detection"""
    print("=== Webcam Face Detection Example ===")
    
    detector = WebcamFaceDetector(camera_id=0)
    detector.start(display=True)


if __name__ == "__main__":
    print("Eye Fatigue Detection - Examples\n")
    
    # Choose example to run
    examples = {
        "1": ("Face Detection", example_face_detection),
        "2": ("Landmark Detection", example_landmark_detection),
        "3": ("Eye Fatigue Detection", example_eye_fatigue),
        "4": ("Webcam Detection", example_webcam_detection),
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    if choice in examples:
        _, example_func = examples[choice]
        example_func()
    else:
        print("Invalid choice")
