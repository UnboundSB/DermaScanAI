import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# Raw LFW directory (Input)
RAW_DIR = r"D:\Projects\DermaScanAI\datasets\facial_recognition\raw\lfw-deepfunneled"

# Final Clean directory (Output)
FINAL_DIR = r"D:\Projects\DermaScanAI\datasets\facial_recognition\final\images_160_png"

# Where to save the plots
EDA_DIR = r"D:\Projects\DermaScanAI\datasets\facial_recognition\final\eda_reports"

# Settings
TARGET_SIZE = (160, 160) # Native for FaceNet
EXTENSION = ".png"       # Lossless

def calculate_brightness_contrast(img):
    """
    Calculates average brightness and contrast (RMS contrast) of an image.
    """
    # Convert to grayscale for calculations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Brightness = Mean pixel value
    brightness = np.mean(gray)
    
    # Contrast = Standard Deviation of pixel values (RMS Contrast)
    contrast = np.std(gray)
    
    return brightness, contrast

def process_single_image(file_info):
    """
    1. Reads Raw Image
    2. Center Crops (Smart)
    3. Resizes to 160x160
    4. Saves as PNG
    5. Returns Brightness/Contrast stats
    """
    root, filename = file_info
    
    src_path = os.path.join(root, filename)
    person_name = os.path.basename(root)
    
    # Construct Destination Path
    dest_folder = os.path.join(FINAL_DIR, person_name)
    fname_no_ext = os.path.splitext(filename)[0]
    dest_path = os.path.join(dest_folder, fname_no_ext + EXTENSION)
    
    # Ensure folder exists
    try:
        os.makedirs(dest_folder, exist_ok=True)
        
        # Load Image
        img = cv2.imread(src_path)
        if img is None:
            return None

        h, w, _ = img.shape

        # --- SMART CENTER CROP (LFW Specific) ---
        # LFW-DeepFunneled is 250x250 with face in center.
        # We crop a 180x180 box from center to remove background noise.
        crop_dim = 180 
        center_y, center_x = h // 2, w // 2
        
        x1 = max(0, center_x - (crop_dim // 2))
        y1 = max(0, center_y - (crop_dim // 2))
        x2 = min(w, center_x + (crop_dim // 2))
        y2 = min(h, center_y + (crop_dim // 2))
        
        crop = img[y1:y2, x1:x2]
        
        # Resize to 160x160 (Standard)
        final_img = cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save as PNG
        cv2.imwrite(dest_path, final_img)
        
        # --- CALCULATE STATS ---
        b, c = calculate_brightness_contrast(final_img)
        
        return {
            "filename": dest_path,
            "person": person_name,
            "brightness": b,
            "contrast": c
        }

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def run_pipeline():
    print(f"--- STARTING PIPELINE ---")
    print(f"Source: {RAW_DIR}")
    print(f"Target: {FINAL_DIR}")
    
    if not os.path.exists(RAW_DIR):
        print("CRITICAL ERROR: Raw directory not found.")
        return

    # 1. Gather Files
    all_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg')):
                all_files.append((root, f))
    
    print(f"Found {len(all_files)} images to process.")

    # 2. Process & Gather Stats (Multithreaded)
    stats_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_single_image, all_files), total=len(all_files), unit="img"))
        
        # Filter None values (errors)
        stats_list = [r for r in results if r is not None]

    print(f"Successfully processed {len(stats_list)} images.")

    # 3. Generate EDA Plots
    print("--- GENERATING EDA PLOTS ---")
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)
        
    df = pd.DataFrame(stats_list)
    
    # Set Plot Style
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Brightness Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['brightness'], bins=50, kde=True, color="orange")
    plt.axvline(x=df['brightness'].mean(), color='red', linestyle='--', label=f"Mean: {df['brightness'].mean():.1f}")
    plt.title("Pixel Brightness Distribution (0=Black, 255=White)")
    plt.xlabel("Average Brightness")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "1_brightness_distro.png"))
    plt.close()
    print("-> Saved 1_brightness_distro.png")

    # PLOT 2: Contrast Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['contrast'], bins=50, kde=True, color="purple")
    plt.axvline(x=df['contrast'].mean(), color='red', linestyle='--', label=f"Mean: {df['contrast'].mean():.1f}")
    plt.title("Image Contrast Distribution (RMS)")
    plt.xlabel("Contrast Level")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "2_contrast_distro.png"))
    plt.close()
    print("-> Saved 2_contrast_distro.png")

    # PLOT 3: Brightness vs Contrast Scatter
    # This helps find "Bad Images" (Low brightness AND Low contrast = useless dark blob)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='brightness', y='contrast', alpha=0.3, s=10)
    plt.title("Brightness vs Contrast Check")
    plt.xlabel("Brightness")
    plt.ylabel("Contrast")
    plt.axvline(x=40, color='r', linestyle=':', label='Too Dark Zone')
    plt.axhline(y=20, color='r', linestyle=':', label='Low Contrast Zone')
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "3_quality_scatter.png"))
    plt.close()
    print("-> Saved 3_quality_scatter.png")

    print(f"--- COMPLETE ---")
    print(f"EDA Reports are in: {EDA_DIR}")

if __name__ == "__main__":
    run_pipeline()