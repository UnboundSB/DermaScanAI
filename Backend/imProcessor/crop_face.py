import os
import sys
import cv2
import numpy as np
import torch

# --- DYNAMIC PATH INJECTION ---
# Ensures Python can find your Backend module
PROJECT_ROOT = r"D:\Projects\DermaScanAI"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Backend.Model.detection.detector import FaceDetector

class FaceNotFoundError(Exception):
    def __init__(self):
        super().__init__("No face Was Detected in the Given Image")
# --- CONFIGURATION ---
DETECTOR_WEIGHTS = r"D:\Projects\DermaScanAI\Backend\Model\detection\face_detector_final.pth"
TEST_IMAGE_PATH = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260224_22_18_22_Pro.jpg"

def crop_primary_face(image_source, detector):
    """
    Receives an image (path or numpy array), detects faces, 
    and returns the cropped numpy array of the highest-confidence face.
    """
    # 1. Handle Input
    if isinstance(image_source, str):
        img_bgr = cv2.imread(image_source)
        if img_bgr is None:
            raise ValueError(f"[!] Error: Could not load image from {image_source}")
    elif isinstance(image_source, np.ndarray):
        img_bgr = image_source
    else:
        raise TypeError("[!] Input must be a file path string or numpy array.")

    # 2. Detect Faces
    faces = detector.detect(img_bgr)
    
    if not faces:
        print("[!] No face detected by the SSDLite model.")
        return None

    # 3. Isolate the most confident face
    best_face = sorted(faces, key=lambda f: f['score'], reverse=True)[0]
    x1, y1, x2, y2 = best_face['box']
    score = best_face['score']
    
    print(f"[*] Target locked. Confidence: {score*100:.1f}%. Coordinates: ({x1}, {y1}) to ({x2}, {y2})")

    # 4. Enforce Boundary Safety
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # 5. Crop and Return
    cropped_face_np = img_bgr[y1:y2, x1:x2]
    return cropped_face_np

if __name__ == "__main__":
    print("--- TESTING FACE CROPPER MODULE ---")
    
    # Initialize the detector ONLY
    print("Loading MobileNetV3 SSDLite Detector...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = FaceDetector(model_path=DETECTOR_WEIGHTS, confidence_threshold=0.5, device=device)

    # Execute the crop
    print(f"Feeding image: {TEST_IMAGE_PATH}")
    cropped_result = crop_primary_face(TEST_IMAGE_PATH, detector)

    if cropped_result is not None:
        # Display the result to prove the crop worked perfectly
        cv2.imshow("Isolated Face Crop", cropped_result)
        print("\n[SUCCESS] Cropper working perfectly. Press any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n[FAILED] Cropper could not return a face.")