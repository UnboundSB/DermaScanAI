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
AUGMENTED_DATA_DIR = os.path.join(BASE_DIR, "dataset_augmented_224_png")
EDA_PLOT_DIR = os.path.join(BASE_DIR, "Augmented_EDA_Reports")

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

def analyze_processed_image(file_path, class_name):
    """Reads the preprocessed PNG and extracts clinical metrics."""
    try:
        img_color = cv2.imread(file_path)
        if img_color is None:
            return None
            
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(img_gray)
        contrast = np.std(img_gray)
        sharpness = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        
        img_lab = cv2.cvtColor(img_color, cv2.COLOR_BGR2LAB)
        skin_tone_ita = calculate_ita_skin_tone(img_lab)

        return {
            "Filename": os.path.basename(file_path),
            "Class": class_name,
            "Brightness": brightness,
            "Contrast": contrast,
            "Sharpness": sharpness,
            "Skin_Tone_ITA": skin_tone_ita
        }
    except Exception as e:
        return None

def main():
    print("--- INITIALIZING FINAL AUGMENTED DATASET AUDIT ---")
    
    if not os.path.exists(AUGMENTED_DATA_DIR):
        print(f"[!] Critical Error: Augmented data directory not found at {AUGMENTED_DATA_DIR}")
        return
        
    os.makedirs(EDA_PLOT_DIR, exist_ok=True)
    
    classes = [d for d in os.listdir(AUGMENTED_DATA_DIR) if os.path.isdir(os.path.join(AUGMENTED_DATA_DIR, d))]
    
    image_paths = []
    for cls in classes:
        cls_dir = os.path.join(AUGMENTED_DATA_DIR, cls)
        for f in os.listdir(cls_dir):
            if f.lower().endswith('.png'):
                image_paths.append((os.path.join(cls_dir, f), cls))
                
    print(f"Found {len(image_paths)} augmented PNG images. Extracting final metrics...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze_processed_image, path, cls): path for path, cls in image_paths}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Auditing"):
            res = future.result()
            if res:
                results.append(res)
                
    df = pd.DataFrame(results)
    
    # --- TERMINAL REPORT ---
    print("\n" + "="*80)
    print(" AUGMENTED DATASET HEALTH REPORT ")
    print("="*80)
    
    print("\n--- FINAL CLASS DISTRIBUTION ---")
    counts = df['Class'].value_counts()
    for cls, count in counts.items():
        print(f" {cls}: {count} images")
        
    print("\n--- CLINICAL AVERAGES PER CLASS (POST-AUGMENTATION) ---")
    metrics_summary = df.groupby('Class')[['Skin_Tone_ITA', 'Brightness', 'Contrast', 'Sharpness']].mean()
    print(metrics_summary.round(2))

    # --- PLOT GENERATION ---
    print("\nRendering augmented clinical distribution maps...")
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Class', hue='Class', order=df['Class'].value_counts().index, palette='viridis', legend=False)
    plt.title('Augmented Images per Symptom Class', fontweight='bold')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "1_augmented_class_distribution.png"))
    plt.close()

    # 2. Skin Tone (ITA) Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Skin_Tone_ITA', hue='Class', fill=True, common_norm=False, palette='Set2')
    plt.title('Augmented Skin Tone Distribution (ITA Angle)', fontweight='bold')
    plt.xlabel('ITA Score (Higher = Lighter Skin, Lower = Darker Skin)')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "2_augmented_skin_tone.png"))
    plt.close()

    # 3. Brightness vs Contrast Map
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Brightness', y='Contrast', hue='Class', alpha=0.5, palette='tab10')
    plt.title('Augmented Lighting Map: Brightness vs Contrast', fontweight='bold')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "3_augmented_lighting_map.png"))
    plt.close()

    # 4. Sharpness Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Class', y='Sharpness', hue='Class', showfliers=False, palette='mako', legend=False)
    plt.title('Augmented Texture Variance (Sharpness)', fontweight='bold')
    plt.ylabel('Laplacian Variance')
    plt.savefig(os.path.join(EDA_PLOT_DIR, "4_augmented_sharpness.png"))
    plt.close()
    
    csv_path = os.path.join(EDA_PLOT_DIR, "augmented_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print("="*80)
    print(f"Audit Complete. Final CSV and plots saved to: {EDA_PLOT_DIR}")

if __name__ == "__main__":
    main()