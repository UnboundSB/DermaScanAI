import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F

# --- CONFIGURATION ---
MODEL_PATH = r"D:\Projects\DermaScanAI\Backend\Model\detection\face_detector_final.pth"
TEST_IMG_DIR = r"D:\Projects\DermaScanAI\datasets\face_detection\processed_640\images" # We test on known data first
OUTPUT_PLOT_PATH = r"D:\Projects\DermaScanAI\Backend\Model\detection\test_results.png"

# Settings
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CONFIDENCE_THRESHOLD = 0.5  # Only show boxes with >50% confidence

def load_model():
    print(f"[Init] Loading model from {MODEL_PATH}...")
    
    # 1. Define Architecture (Must match training exactly)
    # We use the same base weights structure but override the head
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
    
    # 2. Modify Head for 2 Classes (Background + Face)
    # Note: If your training script used a different number, this will crash. 
    # Standard SSDLite usually expects background(0) + face(1) = 2 classes.
    model.head.classification_head.num_classes = 2
    
    # 3. Load Trained Weights
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval() # Set to Evaluation Mode (Freezes Dropout/BatchNorm)
        print("[Success] Model loaded.")
        return model
    else:
        print(f"[Error] Model file not found at {MODEL_PATH}")
        exit()

def run_test(model):
    print("[Test] Running inference on 4 random images...")
    
    # Get 4 random images
    all_imgs = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.png')]
    if not all_imgs:
        print("[Error] No images found in test directory.")
        return
        
    test_samples = np.random.choice(all_imgs, 4, replace=False)
    
    # Create Plot Figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, img_name in enumerate(test_samples):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        # Load & Preprocess
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(original_img).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            # Model expects a list of tensors
            predictions = model([img_tensor])
            
        # Parse Results (Output is a list of dicts)
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        # Draw on Image
        # We need a copy to draw on
        draw_img = original_img.copy()
        
        found_faces = 0
        for box, score in zip(boxes, scores):
            if score > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.astype(int)
                
                # Draw Box (Green)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label
                label = f"{score:.2f}"
                cv2.putText(draw_img, label, (x1, y1-10), cv2.LINE_AA, 0.5, (0, 255, 0), 1)
                found_faces += 1
                
        # Plot
        axes[i].imshow(draw_img)
        axes[i].set_title(f"Faces: {found_faces}")
        axes[i].axis('off')

    # Save Results
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    print(f"[Success] Test plot saved to: {OUTPUT_PLOT_PATH}")
    plt.close()

if __name__ == "__main__":
    detector = load_model()
    run_test(detector)