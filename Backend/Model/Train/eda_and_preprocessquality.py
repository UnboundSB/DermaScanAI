import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# Raw Data Paths
BASE_RAW_DIR = r"D:\Projects\DermaScanAI\datasets\quality\raw\gfiqa-20k\GFIQA-20k"
SOURCE_IMG_DIR = os.path.join(BASE_RAW_DIR, "image")
CSV_PATH = os.path.join(BASE_RAW_DIR, "mos_val_rating.csv")

# Processed Data Paths
BASE_PROC_DIR = r"D:\Projects\DermaScanAI\datasets\quality\processed"
DEST_IMG_DIR = os.path.join(BASE_PROC_DIR, "gfiqa-224-png", "image")
EDA_DIR = os.path.join(BASE_PROC_DIR, "gfiqa-224-png", "eda")

# Settings
TARGET_SIZE = (224, 224)
EXTENSION = ".png" # PNG is lossless (better for IQA than JPG)

# --- PART 1: PREPROCESSING ---
def process_single_image(filename):
    """Reads, resizes, and saves a single image."""
    src_path = os.path.join(SOURCE_IMG_DIR, filename)
    
    # Change extension to .png for destination
    file_root = os.path.splitext(filename)[0]
    dst_filename = file_root + EXTENSION
    dst_path = os.path.join(DEST_IMG_DIR, dst_filename)

    if os.path.exists(dst_path):
        return None # Skip if done

    try:
        img = cv2.imread(src_path)
        if img is None:
            return None

        # Resize using INTER_AREA (best for shrinking images)
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save as PNG
        cv2.imwrite(dst_path, img_resized)
        return dst_filename
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def run_preprocessing():
    if not os.path.exists(DEST_IMG_DIR):
        os.makedirs(DEST_IMG_DIR)
        
    print(f"--- STARTING PREPROCESSING ---")
    print(f"Source: {SOURCE_IMG_DIR}")
    print(f"Dest:   {DEST_IMG_DIR}")

    # Get list of files
    files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Multithreaded processing for speed
    print(f"Resizing {len(files)} images to {TARGET_SIZE}...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_single_image, files), total=len(files), unit="img"))
    
    print("Preprocessing Complete.\n")

# --- PART 2: EDA & PLOTTING ---
def run_eda():
    if not os.path.exists(EDA_DIR):
        os.makedirs(EDA_DIR)
    
    print(f"--- STARTING EDA ---")
    print(f"Saving plots to: {EDA_DIR}")

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH)
        # Rename columns for consistency (adjust indices if your CSV is different)
        df.columns.values[0] = "filename"
        df.columns.values[1] = "mos"
        print(f"Loaded CSV with {len(df)} records.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Set style
    sns.set_theme(style="whitegrid")

    # PLOT 1: MOS Score Distribution (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="mos", bins=30, kde=True, color="skyblue")
    plt.axvline(x=5.0, color='r', linestyle='--', label='Threshold (5.0)')
    plt.title("Distribution of Quality Scores (MOS)")
    plt.xlabel("Mean Opinion Score (0-10)")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(os.path.join(EDA_DIR, "1_mos_distribution.png"))
    plt.close()
    print("-> Saved 1_mos_distribution.png")

    # PLOT 2: Pass/Fail Ratio
    plt.figure(figsize=(6, 6))
    df['status'] = df['mos'].apply(lambda x: 'PASS (>=5)' if x >= 5 else 'FAIL (<5)')
    status_counts = df['status'].value_counts()
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
    plt.title("Gatekeeper Simulation: Pass vs Fail Ratio")
    plt.savefig(os.path.join(EDA_DIR, "2_pass_fail_ratio.png"))
    plt.close()
    print("-> Saved 2_pass_fail_ratio.png")

    # PLOT 3: Sample Images (Low vs High Quality)
    # We need to read from the PROCESSED directory to show what the model actually sees
    print("Generating sample grid (this takes a moment)...")
    
    # Get top 5 best and top 5 worst
    df_sorted = df.sort_values(by="mos")
    worst_files = df_sorted.head(5)['filename'].tolist()
    best_files  = df_sorted.tail(5)['filename'].tolist()
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Visual Comparison: Worst vs Best Quality", fontsize=16)

    def show_img_on_ax(filename, ax, title):
        # Handle extension change (.jpg -> .png)
        base = os.path.splitext(filename)[0]
        path = os.path.join(DEST_IMG_DIR, base + ".png")
        
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "Not Found", ha='center')
            ax.axis('off')

    # Plot Worst
    for i, fname in enumerate(worst_files):
        score = df[df['filename']==fname]['mos'].values[0]
        show_img_on_ax(fname, axes[0, i], f"Score: {score:.2f}")
    axes[0,0].set_ylabel("WORST", fontsize=14, rotation=90)

    # Plot Best
    for i, fname in enumerate(best_files):
        score = df[df['filename']==fname]['mos'].values[0]
        show_img_on_ax(fname, axes[1, i], f"Score: {score:.2f}")
    axes[1,0].set_ylabel("BEST", fontsize=14, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "3_sample_comparison.png"))
    plt.close()
    print("-> Saved 3_sample_comparison.png")

    print(f"--- EDA COMPLETE ---")

if __name__ == "__main__":
    # Ensure raw directory exists before running
    if os.path.exists(SOURCE_IMG_DIR):
        run_preprocessing()
        run_eda()
    else:
        print(f"CRITICAL ERROR: Source directory not found: {SOURCE_IMG_DIR}")