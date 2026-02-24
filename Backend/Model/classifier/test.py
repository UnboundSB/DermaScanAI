import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

# --- CONFIGURATION ---
MODEL_PATH = r"D:\Projects\DermaScanAI\Backend\Model\classifier\symptom_classifier_final4.pth"
IMAGE_PATH = r"C:\Users\dell\Pictures\Camera Roll\WIN_20260206_11_21_11_Pro.jpg"

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

def main():
    print(f"--- INITIATING LIVE DIAGNOSTIC ON {DEVICE.type.upper()} ---")

    if not os.path.exists(IMAGE_PATH):
        print(f"[!] Error: Image not found at {IMAGE_PATH}")
        return

    # 1. LOAD THE BRAIN
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 2. PRE-PROCESS THE CAPTURE
    raw_img = cv2.imread(IMAGE_PATH)
    if raw_img is None:
        print("[!] Error: Could not read image file.")
        return

    # Step A: Apply CLAHE (Clinical Standard)
    normalized_img = apply_clinical_normalization(raw_img)
    
    # Step B: Convert to PIL and apply Grayscale + Tensor transforms
    # We use 3 output channels to satisfy EfficientNet's expected input shape
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(normalized_img).unsqueeze(0).to(DEVICE)

    # 3. RUN INFERENCE
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    # 4. OUTPUT RESULTS
    predicted_class = CLASSES[predicted_idx.item()]
    conf_score = confidence.item() * 100

    print("\n" + "="*40)
    print(f" DIAGNOSTIC RESULT ")
    print("="*40)
    print(f" IMAGE: {os.path.basename(IMAGE_PATH)}")
    print(f" PREDICTION: {predicted_class.upper()}")
    print(f" CONFIDENCE: {conf_score:.2f}%")
    print("="*40)

    # Display the normalized image for visual verification
    display_img = cv2.resize(normalized_img, (600, 600))
    cv2.putText(display_img, f"{predicted_class} ({conf_score:.1f}%)", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("DermaScanAI - Clinical View", display_img)
    print("\nPress any key on the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()