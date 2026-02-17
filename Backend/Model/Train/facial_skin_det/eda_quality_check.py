import os
import sys
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 1. DYNAMIC IMPORT SETUP ---
# Get the current directory (Backend/Model/Train/facial_skin_det)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels to reach 'Backend/Model'
model_root = os.path.abspath(os.path.join(current_dir, "../../"))

# Add this path to Python so we can import 'quality'
if model_root not in sys.path:
    sys.path.append(model_root)

try:
    # Now we can import from the sibling directory 'quality'
    from quality.model import IQAModel
    print(f"[Init] Successfully imported IQAModel from {model_root}\\quality")
except ImportError as e:
    print(f"[Critical Error] Failed to import IQAModel: {e}")
    print(f"Verified search path: {model_root}")
    sys.exit()

# --- CONFIGURATION ---
DATASET_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\dataset_final"
CLASSES = ["clear_face", "darkspots", "puffy_eyes", "wrinkles"]
OUTPUT_REPORT_DIR = os.path.join(DATASET_DIR, "_eda_reports")

# Settings
QUALITY_THRESHOLD = 0.5   # 0.0 to 1.0 (Adjust based on your model's output)
SKIN_TONE_THRESHOLD = 130 # 0-255 (Lower = Darker, Higher = Lighter)

def estimate_skin_tone(image):
    """
    Estimates 'Dark' or 'Light' skin based on the median brightness 
    of the center of the image (to avoid background noise).
    """
    h, w = image.shape[:2]
    # Crop center 30%
    cy, cx = h // 2, w // 2
    h_crop, w_crop = int(h * 0.3), int(w * 0.3)
    crop = image[cy-h_crop:cy+h_crop, cx-w_crop:cx+w_crop]
    
    if crop.size == 0: return "Light" # Fallback

    # Convert to LAB (L = Lightness)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    
    # Use Median to ignore bright reflections (oily skin) or dark shadows
    avg_brightness = np.median(l_channel)
    
    return "Dark" if avg_brightness < SKIN_TONE_THRESHOLD else "Light"

def generate_plots(dark_counts, light_counts, quality_scores):
    print("--- GENERATING DIAGNOSTIC PLOTS ---")
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: BIAS BALANCE (The 45-Degree Line) ---
    plt.figure(figsize=(8, 8))
    
    max_val = 0
    for cls in CLASSES:
        d = dark_counts[cls]
        l = light_counts[cls]
        max_val = max(max_val, d, l)
        
        # Scatter point
        plt.scatter(d, l, s=150, alpha=0.7, label=cls)
        plt.text(d + (max_val*0.02), l, cls, fontsize=10, weight='bold')

    # The Perfect Balance Line
    limit = max_val + 50
    plt.plot([0, limit], [0, limit], 'r--', alpha=0.5, label="Perfect Balance (1:1)")
    
    plt.title("Skin Tone Bias Audit (Dark vs Light Count)")
    plt.xlabel("Count of Dark Skin Images")
    plt.ylabel("Count of Light Skin Images")
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_REPORT_DIR, "1_bias_audit.png")
    plt.savefig(save_path)
    plt.close()

    # --- PLOT 2: QUALITY DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    
    # Flatten scores list
    all_scores = []
    for cls in CLASSES:
        all_scores.extend(quality_scores[cls])
        
    if all_scores:
        sns.histplot(all_scores, bins=40, kde=True, color="teal")
        plt.axvline(x=QUALITY_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f"Reject Threshold ({QUALITY_THRESHOLD})")
        plt.title("Dataset Quality Score Distribution")
        plt.xlabel("Quality Score (Higher is Better)")
        plt.legend()
        
        save_path = os.path.join(OUTPUT_REPORT_DIR, "2_quality_audit.png")
        plt.savefig(save_path)
        plt.close()

    print(f"[Success] Plots saved to: {OUTPUT_REPORT_DIR}")

def run_pipeline():
    print(f"--- STARTING AUDIT ON {DATASET_DIR} ---")
    
    # Init Model
    try:
        iqa = IQAModel()
        print("[Info] IQAModel loaded.")
    except Exception as e:
        print(f"[Error] Model Init Failed: {e}")
        return

    # Prepare Stats
    dark_counts = {c: 0 for c in CLASSES}
    light_counts = {c: 0 for c in CLASSES}
    quality_scores = {c: [] for c in CLASSES}
    
    # Prepare Reject Folder
    reject_dir = os.path.join(DATASET_DIR, "_rejected")
    os.makedirs(reject_dir, exist_ok=True)

    # Scan Loop
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"[Skip] Folder not found: {cls}")
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Scanning {cls} ({len(images)} images)...")

        for img_name in tqdm(images):
            img_path = os.path.join(cls_dir, img_name)
            
            # Load
            img = cv2.imread(img_path)
            if img is None: continue

            # 1. Check Quality
            try:
                # Predict
                q_score = iqa.predict(img_path)
                
                # Handle return types (tensor/list/float)
                if hasattr(q_score, 'item'): q_score = q_score.item()
                if isinstance(q_score, list): q_score = q_score[0]
            except:
                print(f"[Warn] Prediction failed for {img_name}")
                q_score = 0.0

            quality_scores[cls].append(q_score)

            # 2. Reject Low Quality
            if q_score < QUALITY_THRESHOLD:
                # Move to rejected
                dest_folder = os.path.join(reject_dir, cls)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.move(img_path, os.path.join(dest_folder, img_name))
                continue # Skip bias check for trash images

            # 3. Check Bias (Only for kept images)
            tone = estimate_skin_tone(img)
            if tone == "Dark":
                dark_counts[cls] += 1
            else:
                light_counts[cls] += 1

    # Generate Reports
    generate_plots(dark_counts, light_counts, quality_scores)
    
    print("\n--- FINAL BIAS SUMMARY ---")
    for cls in CLASSES:
        d = dark_counts[cls]
        l = light_counts[cls]
        print(f"{cls.ljust(15)}: {d} Dark | {l} Light")
        
        if d == 0 and l > 0:
             print(f"   >>> CRITICAL WARNING: {cls} has ZERO dark skin samples. Model will be biased.")

if __name__ == "__main__":
    run_pipeline()