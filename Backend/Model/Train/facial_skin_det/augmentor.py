import os
import cv2
import math
import shutil
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
SOURCE_DIR = os.path.join(BASE_DIR, "dataset_processed_224_png")
AUGMENTED_DIR = os.path.join(BASE_DIR, "dataset_augmented_224_png")

TARGET_MIN_IMAGES = 900
EXCLUDED_CLASSES = ['puffy_eyes']

# Strict Spatial Augmentation (Preserves original lighting, sharpness, and color distributions)
geometric_augmenter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
])

def calculate_ita_skin_tone(img_color):
    """Calculates Individual Typology Angle (ITA) for clinical skin tone estimation."""
    # Convert BGR to CIELAB color space
    img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)
    
    # OpenCV scales LAB differently, so we normalize back to standard CIELAB ranges
    L_true = (L.astype(np.float32) * 100.0) / 255.0
    b_true = b.astype(np.float32) - 128.0
    
    # We sample the central 50% of the image to target the facial skin, ignoring backgrounds
    h, w = L_true.shape
    c_h1, c_h2 = int(h * 0.25), int(h * 0.75)
    c_w1, c_w2 = int(w * 0.25), int(w * 0.75)
    
    center_L = np.mean(L_true[c_h1:c_h2, c_w1:c_w2])
    center_b = np.mean(b_true[c_h1:c_h2, c_w1:c_w2])
    
    # Prevent division by zero
    b_val = center_b if center_b != 0 else 0.001
    ita_score = math.atan((center_L - 50.0) / b_val) * (180.0 / math.pi)
    
    return ita_score

def augment_image(src_path, dest_path):
    """Applies strict spatial augmentation and saves the synthetic image."""
    img_bgr = cv2.imread(src_path)
    if img_bgr is None: return
    
    # Convert to PIL for PyTorch transforms (requires RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Apply spatial shifts
    aug_pil = geometric_augmenter(pil_img)
    
    # Convert back to OpenCV BGR and save
    aug_bgr = cv2.cvtColor(np.array(aug_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(dest_path, aug_bgr)

def main():
    print("--- INITIALIZING STRATIFIED AUGMENTATION ENGINE ---")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"[!] Critical Error: Source directory not found at {SOURCE_DIR}")
        return
        
    # Zero-Trust Wipe of Augmented Directory
    if os.path.exists(AUGMENTED_DIR):
        print("[*] Wiping previous augmented directory to prevent ghost files...")
        shutil.rmtree(AUGMENTED_DIR)
        
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    for cls in classes:
        src_cls_dir = os.path.join(SOURCE_DIR, cls)
        dest_cls_dir = os.path.join(AUGMENTED_DIR, cls)
        os.makedirs(dest_cls_dir, exist_ok=True)
        
        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith('.png')]
        current_count = len(files)
        
        print(f"\nProcessing '{cls}' (Current Count: {current_count})")
        
        # Step 1: Lock in the Ground Truth
        for f in tqdm(files, desc="Copying Ground Truth", leave=False):
            shutil.copy2(os.path.join(src_cls_dir, f), os.path.join(dest_cls_dir, f))
            
        # Step 2: Verification Checks
        if cls in EXCLUDED_CLASSES:
            print(f"> Class '{cls}' is excluded. Skipping synthesis.")
            continue
            
        if current_count >= TARGET_MIN_IMAGES:
            print(f"> Class '{cls}' meets minimum target. Skipping synthesis.")
            continue
            
        # Step 3: Stratify by Skin Tone (ITA)
        needed_images = TARGET_MIN_IMAGES - current_count
        print(f"> Synthesizing {needed_images} images to reach {TARGET_MIN_IMAGES} target...")
        
        ita_records = []
        for f in files:
            src_path = os.path.join(src_cls_dir, f)
            img_color = cv2.imread(src_path)
            if img_color is not None:
                ita_val = calculate_ita_skin_tone(img_color)
                ita_records.append((f, ita_val))
                
        # Sort by ITA score (lowest to highest)
        ita_records.sort(key=lambda x: x[1])
        midpoint = len(ita_records) // 2
        
        # Split into dark and light pools
        dark_skin_pool = [x[0] for x in ita_records[:midpoint]]
        light_skin_pool = [x[0] for x in ita_records[midpoint:]]
        
        # Step 4: Generate Balanced Synthetics
        for i in tqdm(range(needed_images), desc="Generating Synthetics", leave=False):
            if i % 2 == 0:
                base_img = random.choice(dark_skin_pool)
                prefix = "aug_dark_"
            else:
                base_img = random.choice(light_skin_pool)
                prefix = "aug_light_"
                
            src_path = os.path.join(src_cls_dir, base_img)
            dest_filename = f"{prefix}{i:04d}_{base_img}"
            dest_path = os.path.join(dest_cls_dir, dest_filename)
            
            augment_image(src_path, dest_path)
            
    print("\n" + "="*80)
    print(" AUGMENTATION PIPELINE COMPLETE ")
    print("="*80)
    print(f"Dataset successfully compiled at: {AUGMENTED_DIR}")

if __name__ == "__main__":
    main()