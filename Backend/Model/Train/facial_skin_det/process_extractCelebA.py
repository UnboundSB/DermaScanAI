import os
import shutil
import random
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"

# EXACT paths provided by you
CELEBA_IMG_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\CelebA\img_align_celeba\img_align_celeba"
CELEBA_ATTR_FILE = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\CelebA\list_attr_celeba.csv"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "binary_skin_tone_effnetb0.pth")

FINAL_DATASET_DIR = os.path.join(BASE_DIR, "dataset_final")
EXTRAS_DIR = os.path.join(FINAL_DATASET_DIR, "extras_unbalanced")

MAX_TO_SCAN = 4000  # Pool size to prevent scanning all 200k images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

def load_ai_model():
    print(f"Loading EfficientNet-B0 Skin Tone AI on {DEVICE}...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) 
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"[Critical Error] Model not found at {MODEL_PATH}.")
        exit()
        
    model = model.to(DEVICE)
    model.eval()
    return model

def get_brightness(image_path):
    """Calculates basic grayscale brightness for EDA."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return np.mean(img) if img is not None else 0
    except:
        return 0

def process_celeba_unzipped():
    print(f"--- FILTERING CELEBA DATASET (UNZIPPED) ---")
    
    os.makedirs(EXTRAS_DIR, exist_ok=True)
    for folder in ["puffy_eyes", "clear_face"]:
        os.makedirs(os.path.join(FINAL_DATASET_DIR, folder), exist_ok=True)

    if not os.path.exists(CELEBA_IMG_DIR) or not os.path.exists(CELEBA_ATTR_FILE):
        print(f"[Error] Could not find CelebA images or CSV. Check paths.")
        return

    # 1. Load Attributes
    print("Loading CelebA CSV attributes into Pandas...")
    df = pd.read_csv(CELEBA_ATTR_FILE)
    
    # Identify the image filename column (usually 'image_id' in Kaggle CSVs)
    img_col = 'image_id' if 'image_id' in df.columns else df.columns[0]
    
    # 2. Filter Logic (Using <= 0 to safely catch both -1 and 0 false formats)
    print("Applying physical trait filters...")
    puffy_candidates = df[df['Bags_Under_Eyes'] == 1][img_col].tolist()
    
    clear_candidates = df[
        (df['Bags_Under_Eyes'] <= 0) & 
        (df['Young'] == 1) & 
        (df['Heavy_Makeup'] <= 0) & 
        (df['No_Beard'] == 1) & 
        (df['Eyeglasses'] <= 0)
    ][img_col].tolist()

    random.shuffle(puffy_candidates)
    random.shuffle(clear_candidates)
    
    candidates = {
        "puffy_eyes": puffy_candidates[:MAX_TO_SCAN],
        "clear_face": clear_candidates[:MAX_TO_SCAN]
    }

    # 3. AI Initialization
    model = load_ai_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    master_stats = {}
    master_dark_brightness = []
    master_light_brightness = []
    master_logits = []

    # 4. Scanning & Balancing
    for cls, filenames in candidates.items():
        print(f"\n--- AI Sorting '{cls}' ({len(filenames)} candidates) ---")
        dark_images, light_images = [], []
        
        for filename in tqdm(filenames, desc=f"Scoring {cls}"):
            img_path = os.path.join(CELEBA_IMG_DIR, filename)
            if not os.path.exists(img_path): continue
            
            try:
                pil_img = Image.open(img_path).convert('RGB')
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logit = model(input_tensor).item()
                
                master_logits.append(logit)
                brightness = get_brightness(img_path)

                if logit > 0:
                    dark_images.append(img_path)
                    master_dark_brightness.append(brightness)
                else:
                    light_images.append(img_path)
                    master_light_brightness.append(brightness)
            except:
                pass 

        orig_dark = len(dark_images)
        orig_light = len(light_images)
        
        target_count = min(orig_dark, orig_light)
        print(f"[{cls}] Found: {orig_dark} Dark vs {orig_light} Light. Forcing 1:1 balance at {target_count} each.")
        
        if target_count == 0:
            print(f"[{cls}] Failed to balance (one category is empty).")
            continue
            
        selected_dark = set(random.sample(dark_images, target_count))
        selected_light = set(random.sample(light_images, target_count))
        
        dest_dir = os.path.join(FINAL_DATASET_DIR, cls)
        added_count = 0
        
        for img_list, tone in [(dark_images, "dark"), (light_images, "light")]:
            for img_path in img_list:
                _, ext = os.path.splitext(img_path)
                safe_name = f"celeba_{cls}_{tone}_{added_count:05d}{ext.lower()}"
                
                if img_path in (selected_dark | selected_light):
                    shutil.copy(img_path, os.path.join(dest_dir, safe_name))
                else:
                    shutil.copy(img_path, os.path.join(EXTRAS_DIR, safe_name))
                added_count += 1
                
        master_stats[cls] = {
            "orig_dark": orig_dark, "orig_light": orig_light, "balanced_total": target_count * 2
        }

    # 5. Master EDA Dashboard
    if master_stats:
        print("\nGenerating CelebA 4-Panel Dashboard...")
        sns.set_theme(style="whitegrid")
        classes_processed = list(master_stats.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("CelebA: Puffy Eyes & Clear Face Dataset Audit", fontsize=16, weight='bold')

        # Plot 1: Unbalanced Source
        dark_orig = [master_stats[c]["orig_dark"] for c in classes_processed]
        light_orig = [master_stats[c]["orig_light"] for c in classes_processed]
        x = np.arange(len(classes_processed))
        width = 0.35
        
        axes[0,0].bar(x - width/2, dark_orig, width, label='Dark Skin (Raw)', color='#8D5524')
        axes[0,0].bar(x + width/2, light_orig, width, label='Light Skin (Raw)', color='#FFC0CB')
        axes[0,0].set_title("Original Imbalance in CelebA Extract", fontsize=12)
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(classes_processed)
        axes[0,0].legend()

        # Plot 2: Balanced Final
        balanced_counts = [master_stats[c]["balanced_total"] // 2 for c in classes_processed]
        
        axes[0,1].bar(x - width/2, balanced_counts, width, label='Dark Skin', color='#8D5524')
        axes[0,1].bar(x + width/2, balanced_counts, width, label='Light Skin', color='#FFC0CB')
        axes[0,1].set_title("Safely Added to dataset_final", fontsize=12)
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(classes_processed)
        axes[0,1].legend()

        # Plot 3: Brightness Distribution
        sns.kdeplot(master_dark_brightness, fill=True, color="#8D5524", label="Predicted Dark", ax=axes[1,0])
        sns.kdeplot(master_light_brightness, fill=True, color="#FFC0CB", label="Predicted Light", ax=axes[1,0])
        axes[1,0].set_title("Pixel Brightness vs. AI Prediction", fontsize=12)
        axes[1,0].set_xlabel("Average Brightness (0=Black, 255=White)")
        axes[1,0].legend()

        # Plot 4: AI Confidence (Logits)
        sns.histplot(master_logits, bins=40, kde=True, color="purple", ax=axes[1,1])
        axes[1,1].axvline(x=0, color='red', linestyle='--', label="Decision Boundary")
        axes[1,1].set_title("AI Model Confidence Distribution", fontsize=12)
        axes[1,1].set_xlabel("<- Light Skin | Dark Skin ->")
        axes[1,1].legend()

        plot_path = os.path.join(FINAL_DATASET_DIR, "celeba_comprehensive_audit.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print("\n" + "="*40)
    print("CELEBA PROCESSING COMPLETE")
    print("="*40)
    for cls, stats in master_stats.items():
        print(f"{cls.ljust(12)}: Added {stats['balanced_total']} perfectly balanced images.")

if __name__ == "__main__":
    process_celeba_unzipped()