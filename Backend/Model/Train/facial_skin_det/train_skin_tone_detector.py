import os
import cv2
import math
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

def process_and_analyze_image(src_path, dest_dir, class_name):
    try:
        img_color = cv2.imread(src_path)
        if img_color is None:
            return None
            
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # --- 1. CLINICAL METRIC EXTRACTION ---
        h, w = img_gray.shape
        brightness = np.mean(img_gray)
        contrast = np.std(img_gray)
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        skin_tone_ita = calculate_ita_skin_tone(img_color)
        
        # --- 2. PIPELINE PREPROCESSING ---
        img_resized = cv2.resize(img_color, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        filename = os.path.splitext(os.path.basename(src_path))[0]
        dest_filename = f"{filename}.png"
        dest_path = os.path.join(dest_dir, class_name, dest_filename)
        
        cv2.imwrite(dest_path, img_resized)

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
    print("--- INITIALIZING CLINICAL EDA & PREPROCESSING PIPELINE ---")
    
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[!] Critical Error: Raw data directory not found at {RAW_DATA_DIR}")
        return
        
    os.makedirs(EDA_PLOT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    for cls in classes:
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, cls), exist_ok=True)
        
    image_tasks = []
    for cls in classes:
        cls_dir = os.path.join(RAW_DATA_DIR, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                src_path = os.path.join(cls_dir, f)
                image_tasks.append((src_path, PROCESSED_DATA_DIR, cls))
                
    print(f"Scanning {len(image_tasks)} images for topological and chromatic metrics...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_and_analyze_image, task[0], task[1], task[2]): task for task in image_tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting Metrics"):
            res = future.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # --- TERMINAL REPORT ---
    print("\n" + "="*80)
    print(" CLINICAL DATASET HEALTH REPORT ")
    print("="*80)
    
    print("\n--- CLASS DISTRIBUTION ---")
    counts = df['Class'].value_counts()
    for cls, count in counts.items():
        print(f" {cls}: {count} images")
        
    print("\n--- CLINICAL AVERAGES PER SYMPTOM CLASS ---")
    metrics_summary = df.groupby('Class')[['Skin_Tone_ITA', 'Brightness', 'Contrast', 'Sharpness']].mean()
    print(metrics_summary.round(2))

    # --- PLOT GENERATION ---
    print("\nRendering clinical distribution maps...")
    sns.set_theme(style="whitegrid")
    
    # 1. Skin Tone (ITA) Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Skin_Tone_ITA', hue='Class', fill=True, common_norm=False, palette='Set2')
    plt.title('Clinical Skin Tone Distribution (ITA Angle)', fontweight='bold')
    plt.xlabel('ITA Score (Higher = Lighter Skin, Lower = Darker Skin)')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "1_skin_tone_distribution.png"))
    plt.close()

    # 2. Brightness vs Contrast Map
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Brightness', y='Contrast', hue='Class', alpha=0.5, palette='tab10')
    plt.title('Topological Lighting Map: Brightness vs Contrast', fontweight='bold')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "2_lighting_map.png"))
    plt.close()

    # 3. Sharpness Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Sharpness', showfliers=False, palette='mako')
    plt.title('Texture Bias Check: Sharpness / Blur Variance', fontweight='bold')
    plt.ylabel('Laplacian Variance (Sharpness)')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "3_sharpness_bias.png"))
    plt.close()
    
    csv_path = os.path.join(EDA_PLOT_DIR, "clinical_eda_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print("="*80)
    print(f"Pipeline Complete. PNGs standardardized. Clinical plots saved to: {EDA_PLOT_DIR}")

if __name__ == "__main__":
    main()