import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

# --- IMPORT YOUR CUSTOM FACE DETECTOR ---
# Assuming your function takes a numpy array and returns a cropped BGR numpy array.
# Adjust the import name to match the exact function inside model.detection.model
try:
    from Model.detection.detector import detect_and_crop_face 
except ImportError:
    print("[!] Warning: Could not import your face detector. Ensure your python path is set correctly.")
    # Fallback placeholder to prevent hard crashes during setup
    def detect_and_crop_face(img_np):
        print("[!] Using fallback: Returning uncropped image.")
        return img_np

# --- CONFIGURATION ---
MODEL_PATH = r"D:\Projects\DermaScanAI\Backend\Model\detection\face_detector_final.pth"
TEST_IMAGE_PATH = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260224_22_18_22_Pro.jpg"

CLASSES = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_clinical_normalization(img_bgr):
    """Applies the exact CLAHE normalization used during training."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def load_clinical_brain():
    """Loads the final diagnostic EfficientNet-B0 model."""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def analyze_face(cropped_face_bgr, model):
    """Runs the normalized cropped face through the neural network."""
    # 1. Clinical CLAHE Normalization
    normalized_img = apply_clinical_normalization(cropped_face_bgr)
    
    # 2. PyTorch Preprocessing (matching training conditions)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(normalized_img).unsqueeze(0).to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_class = CLASSES[predicted_idx.item()]
    conf_score = confidence.item() * 100
    
    return predicted_class, conf_score, normalized_img

def main():
    print(f"--- INITIATING END-TO-END DIAGNOSTIC PIPELINE ON {DEVICE.type.upper()} ---")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[!] Error: Test image not found at {TEST_IMAGE_PATH}")
        return
        
    # 1. Load the raw image as a numpy array
    print("[*] Loading raw image into memory...")
    raw_img_np = cv2.imread(TEST_IMAGE_PATH)
    if raw_img_np is None:
        print("[!] Error: OpenCV failed to decode the image.")
        return

    # 2. Extract the face using your custom detector
    print("[*] Engaging facial detection architecture...")
    cropped_face_np = detect_and_crop_face(raw_img_np)
    
    if cropped_face_np is None or cropped_face_np.size == 0:
        print("[!] Neural failure: No face detected in the image.")
        return
        
    print("[*] Face isolated successfully.")

    # 3. Load the diagnostic brain
    print("[*] Loading clinical classifier weights...")
    clinical_model = load_clinical_brain()

    # 4. Analyze the isolated face
    print("[*] Executing clinical topological scan...")
    predicted_class, conf_score, normalized_img = analyze_face(cropped_face_np, clinical_model)

    # 5. Output Telemetry
    print("\n" + "="*40)
    print(" CLINICAL DIAGNOSTIC RESULT ")
    print("="*40)
    print(f" TARGET: {os.path.basename(TEST_IMAGE_PATH)}")
    print(f" DIAGNOSIS: {predicted_class.upper()}")
    print(f" CONFIDENCE: {conf_score:.2f}%")
    print("="*40)

    # 6. Visual Verification
    display_img = cv2.resize(normalized_img, (600, 600))
    cv2.putText(display_img, f"{predicted_class.upper()} ({conf_score:.1f}%)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    cv2.imshow("DermaScanAI - End-to-End Pipeline", display_img)
    print("\nPress any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()