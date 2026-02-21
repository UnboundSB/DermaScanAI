import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
INPUT_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")
REPORT_DIR = os.path.join(BASE_DIR, "EDA_Reports")

CLASSES = ["acne", "darkspots", "wrinkles", "puffy_eyes", "clear_face"]

def calculate_ita(l_channel, b_channel):
    """Calculates Individual Typology Angle (ITA) for skin tone."""
    l_std = l_channel.astype(np.float32) * 100.0 / 255.0
    b_std = b_channel.astype(np.float32) - 128.0
    
    # Avoid division by zero
    b_std[b_std == 0] = 1e-5 
    
    ita = np.arctan2((l_std - 50), b_std) * (180 / np.pi)
    return np.mean(ita)

def get_skin_tone_category(ita_value):
    if ita_value > 55: return "Very Light"
    elif 41 < ita_value <= 55: return "Light"
    elif 28 < ita_value <= 41: return "Intermediate"
    elif 10 < ita_value <= 28: return "Tan"
    elif -30 < ita_value <= 10: return "Brown"
    else: return "Dark"

def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    
    # 1. Create a mask to IGNORE the black padding added during cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    
    if not np.any(mask): return None 
    
    valid_pixels = gray[mask]
    
    # 2. Quality Metrics
    brightness = np.mean(valid_pixels)
    contrast = np.std(valid_pixels)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() 
    
    # 3. Skin Tone (ITA) Analysis
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    l_valid = l_channel[mask]
    b_valid = b_channel[mask]
    
    ita_val = calculate_ita(l_valid, b_valid)
    skin_tone = get_skin_tone_category(ita_val)
    
    return {
        "Brightness": brightness,
        "Contrast": contrast,
        "Sharpness": sharpness,
        "ITA_Value": ita_val,
        "Skin_Tone": skin_tone
    }

def main():
    print("--- STARTING DATASET EXPLORATORY ANALYSIS ---")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    data = []
    
    for cls in CLASSES:
        target_folder = os.path.join(INPUT_DIR, cls)
        if not os.path.exists(target_folder): continue
            
        images = [f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not images: continue
        
        print(f"Analyzing {cls} ({len(images)} images)...")
        
        for img_name in tqdm(images):
            img_path = os.path.join(target_folder, img_name)
            metrics = analyze_image(img_path)
            
            if metrics:
                metrics["Class"] = cls
                data.append(metrics)
                
    if not data:
        print("[Error] No valid images found to analyze.")
        return
        
    df = pd.DataFrame(data)
    
    # ==========================================
    # GENERATING PLOTS
    # ==========================================
    print("\nGenerating Visual Plots...")
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Class", order=CLASSES, palette="viridis")
    plt.title("Final Class Distribution")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(REPORT_DIR, "1_class_distribution.png"))
    plt.close()
    
    # 2. Skin Tone Distribution
    plt.figure(figsize=(12, 6))
    tone_order = ["Very Light", "Light", "Intermediate", "Tan", "Brown", "Dark"]
    sns.countplot(data=df, x="Class", hue="Skin_Tone", hue_order=tone_order, palette="YlOrBr")
    plt.title("Skin Tone Representation per Symptom Class")
    plt.ylabel("Number of Images")
    plt.legend(title="Skin Tone", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "2_skin_tone_distribution.png"))
    plt.close()

    # 3. Image Quality Metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.kdeplot(data=df, x="Brightness", hue="Class", fill=True, ax=axes[0])
    axes[0].set_title("Brightness Distribution")
    
    sns.kdeplot(data=df, x="Contrast", hue="Class", fill=True, ax=axes[1])
    axes[1].set_title("Contrast Distribution")
    
    df['Log_Sharpness'] = np.log1p(df['Sharpness'])
    sns.kdeplot(data=df, x="Log_Sharpness", hue="Class", fill=True, ax=axes[2])
    axes[2].set_title("Sharpness Distribution (Log Scale)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "3_quality_metrics.png"))
    plt.close()

    df.to_csv(os.path.join(REPORT_DIR, "full_eda_metrics.csv"), index=False)
    
    # ==========================================
    # TERMINAL NUMERICAL REPORT
    # ==========================================
    print("\n" + "="*50)
    print("NUMERICAL RESULTS SUMMARY")
    print("="*50)
    
    print("\n--- 1. Class Distribution ---")
    class_counts = df['Class'].value_counts()
    for cls_name, count in class_counts.items():
        print(f"{cls_name:<15}: {count} images")

    print("\n--- 2. Skin Tone Distribution by Class ---")
    # Create a pivot table for clean terminal viewing
    skin_tone_pivot = df.groupby(['Class', 'Skin_Tone']).size().unstack(fill_value=0)
    # Ensure columns follow the logical light-to-dark order if present
    available_tones = [t for t in tone_order if t in skin_tone_pivot.columns]
    skin_tone_pivot = skin_tone_pivot[available_tones]
    print(skin_tone_pivot.to_string())

    print("\n--- 3. Average Quality Metrics by Class ---")
    quality_metrics = df.groupby('Class')[['Brightness', 'Contrast', 'Sharpness']].mean().round(2)
    print(quality_metrics.to_string())
    
    print("\n" + "="*50)
    print("EDA COMPLETE")
    print("="*50)
    print(f"Total Images Analyzed: {len(df)}")
    print(f"All reports and plots saved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()