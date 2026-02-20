import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Safely import YOUR detector from the file sitting right next to this one
from custom_face_model import FaceDetector

# --- CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_WEIGHTS = os.path.join(CURRENT_DIR, "face_detector_final.pth")

BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
INPUT_DIR = os.path.join(BASE_DIR, "dataset_final")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")

CLASSES = ["acne", "darkspots", "wrinkles", "puffy_eyes", "clear_face"]
TARGET_SIZE = 224
MARGIN = 0.2  

def get_best_face_box(detector, img):
    try:
        faces = detector.detect(img)
        if not faces or len(faces) == 0:
            return None
        best_face = max(faces, key=lambda f: f['score'])
        x1, y1, x2, y2 = best_face['box']
        return int(x1), int(y1), int(x2), int(y2)
    except Exception as e:
        return None

def process_image(img_path, detector):
    try:
        img = cv2.imread(img_path)
        if img is None: return None
        
        box = get_best_face_box(detector, img)
        if box is None: return None 
        
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        
        size = max(w, h) * (1 + MARGIN)
        
        new_x1 = max(0, int(center_x - size / 2))
        new_y1 = max(0, int(center_y - size / 2))
        new_x2 = min(img.shape[1], int(center_x + size / 2))
        new_y2 = min(img.shape[0], int(center_y + size / 2))
        
        face_crop = img[new_y1:new_y2, new_x1:new_x2]
        crop_h, crop_w = face_crop.shape[:2]
        
        if crop_h == 0 or crop_w == 0: return None
        
        if crop_h < TARGET_SIZE or crop_w < TARGET_SIZE:
            delta_w = max(0, TARGET_SIZE - crop_w)
            delta_h = max(0, TARGET_SIZE - crop_h)
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            final_img = cv2.copyMakeBorder(face_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            final_img = cv2.resize(face_crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
            
        return final_img
    except Exception as e:
        return None

def main():
    print(f"--- STARTING FACE EXTRACTION USING SSDLITE CUSTOM DETECTOR ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(DETECTOR_WEIGHTS):
        print(f"[Critical Error] Could not find weights at {DETECTOR_WEIGHTS}")
        return

    print("Loading Custom Face Detector...")
    detector = FaceDetector(model_path=DETECTOR_WEIGHTS, confidence_threshold=0.5)
    
    total_processed = 0
    total_failed = 0
    
    for cls in CLASSES:
        source_folder = os.path.join(INPUT_DIR, cls)
        dest_folder = os.path.join(OUTPUT_DIR, cls)
        if not os.path.exists(source_folder): continue
            
        os.makedirs(dest_folder, exist_ok=True)
        images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images: continue
        
        print(f"\nProcessing '{cls}' ({len(images)} images)...")
        success_count = 0
        
        for img_name in tqdm(images):
            img_path = os.path.join(source_folder, img_name)
            final_img = process_image(img_path, detector)
            
            if final_img is not None:
                save_path = os.path.join(dest_folder, img_name)
                cv2.imwrite(save_path, final_img)
                success_count += 1
            else:
                total_failed += 1
                
        total_processed += success_count
        print(f"[{cls}] Saved {success_count} perfect 224x224 crops.")

    print("\n" + "="*40)
    print("CUSTOM RESIZE & CROP PIPELINE COMPLETE")
    print("="*40)
    print(f"Successfully prepped images: {total_processed}")
    print(f"Failed/No Face Detected:     {total_failed}")

if __name__ == "__main__":
    main()