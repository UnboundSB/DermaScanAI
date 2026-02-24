import os
import cv2
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import concurrent.futures

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
RAW_DATA_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset_processed_224_png")
EDA_PLOT_DIR = os.path.join(BASE_DIR, "Clinical_EDA_Reports")

TARGET_SIZE = (224, 224)
TARGET_CLEAR_FACE_COUNT = 600 

def get_image_sharpness(img_path):
    """Calculates Laplacian variance to rank image sharpness."""
    try:
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return 0
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()
    except:
        return 0

def calculate_ita_skin_tone(img_lab):
    """Calculates Individual Typology Angle (ITA) from LAB color space."""
    L, a, b = cv2.split(img_lab)
    L_true = (L.astype(np.float32) * 100.0) / 255.0
    b_true = b.astype(np.float32) - 128.0
    
    h, w = L_true.shape
    c_h1, c_h2 = int(h * 0.25), int(h * 0.75)
    c_w1, c_w2 = int(w * 0.25), int(w * 0.75)
    
    center_L = np.mean(L_true[c_h1:c_h2, c_w1:c_w2])
    center_b = np.mean(b_true[c_h1:c_h2, c_w1:c_w2])
    
    b_val = center_b if center_b != 0 else 0.001
    return math.atan((center_L - 50.0) / b_val) * (180.0 / math.pi)

def process_and_analyze_image(src_path, dest_dir, class_name):
    """Applies CLAHE, resizes, saves as PNG, and extracts final EDA metrics."""
    try:
        img_color = cv2.imread(src_path)
        if img_color is None: return None
        
        # --- 1. CLINICAL NORMALIZATION (CLAHE) ---
        lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_normalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # --- 2. RESIZE & SAVE ---
        img_resized = cv2.resize(img_normalized, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        filename = os.path.splitext(os.path.basename(src_path))[0]
        dest_filename = f"{filename}.png"
        dest_path = os.path.join(dest_dir, class_name, dest_filename)
        cv2.imwrite(dest_path, img_resized)

        # --- 3. EXTRACT POST-PROCESS METRICS ---
        img_gray_final = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(img_gray_final)
        contrast = np.std(img_gray_final)
        sharpness = cv2.Laplacian(img_gray_final, cv2.CV_64F).var()
        skin_tone_ita = calculate_ita_skin_tone(cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB))

        return {
            "Filename": dest_filename,
            "Class": class_name,
            "Brightness": brightness,
            "Contrast": contrast,
            "Sharpness": sharpness,
            "Skin_Tone_ITA": skin_tone_ita
        }
    except Exception as e:
        return None

def main():
    print("--- INITIALIZING SURGICAL PURGE & CLINICAL PIPELINE ---")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[!] Directory not found: {RAW_DATA_DIR}")
        return
        
    os.makedirs(EDA_PLOT_DIR, exist_ok=True)
    
    # --- ZERO-TRUST DIRECTORY WIPE ---
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"[*] ANOMALY DETECTED: Pre-existing processed directory found.")
        print(f"[*] EXECUTING PROTOCOL: Scorched earth. Deleting {PROCESSED_DATA_DIR} to prevent ghost files...")
        shutil.rmtree(PROCESSED_DATA_DIR)
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, cls), exist_ok=True)
        
    final_image_tasks = []
    
    # --- PHASE 1: THE ALGORITHMIC MACHETE ---
    for cls in classes:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if cls == 'clear_face' and len(files) > TARGET_CLEAR_FACE_COUNT:
            print(f"\n[!] Targeting '{cls}' for surgical purge (Found {len(files)} images).")
            print(f"Calculating Laplacian variance to isolate the {TARGET_CLEAR_FACE_COUNT} sharpest images...")
            
            sharpness_scores = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                scores = list(tqdm(executor.map(get_image_sharpness, files), total=len(files), desc="Scanning Sharpness"))
                
            for path, score in zip(files, scores):
                sharpness_scores.append((path, score))
                
            sharpness_scores.sort(key=lambda x: x[1], reverse=True)
            kept_files = [x[0] for x in sharpness_scores[:TARGET_CLEAR_FACE_COUNT]]
            
            print(f"> Purged {len(files) - TARGET_CLEAR_FACE_COUNT} blurry/airbrushed images.")
            for f in kept_files:
                final_image_tasks.append((f, PROCESSED_DATA_DIR, cls))
        else:
            for f in files:
                final_image_tasks.append((f, PROCESSED_DATA_DIR, cls))
                
    # --- PHASE 2: CLINICAL NORMALIZATION & EDA ---
    print(f"\nCommencing CLAHE Normalization and 224x224 conversion on {len(final_image_tasks)} surviving images...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_and_analyze_image, task[0], task[1], task[2]): task for task in final_image_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            res = future.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # --- TERMINAL REPORT ---
    print("\n" + "="*80)
    print(" POST-PURGE DATASET HEALTH REPORT ")
    print("="*80)
    
    print("\n--- NEW CLASS DISTRIBUTION ---")
    counts = df['Class'].value_counts()
    for cls, count in counts.items():
        print(f" {cls}: {count} images")
        
    print("\n--- NEW CLINICAL AVERAGES PER CLASS ---")
    metrics_summary = df.groupby('Class')[['Skin_Tone_ITA', 'Brightness', 'Contrast', 'Sharpness']].mean()
    print(metrics_summary.round(2))

    # --- PLOTS ---
    print("\nRendering new clinical distribution maps...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Brightness', y='Contrast', hue='Class', alpha=0.5, palette='tab10')
    plt.title('Post-CLAHE Lighting Map: Brightness vs Contrast', fontweight='bold')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "2_post_clahe_lighting.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Sharpness', showfliers=False, palette='mako')
    plt.title('Post-Purge Texture Variance', fontweight='bold')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "3_post_purge_sharpness.png"))
    plt.close()
    
    csv_path = os.path.join(EDA_PLOT_DIR, "post_purge_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print("="*80)
    print(f"Pipeline Complete. Clinically clean PNGs saved to: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()