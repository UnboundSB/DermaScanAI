import os
import shutil
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
# Pointing directly to your extracted folder
SOURCE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\ageing other\Skin Issues Dataset\Skin v2"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "binary_skin_tone_effnetb0.pth")

FINAL_DATASET_DIR = os.path.join(BASE_DIR, "dataset_final")
EXTRAS_DIR = os.path.join(FINAL_DATASET_DIR, "extras_unbalanced")

# Classes we want to extract and their target folder names
TARGET_CLASSES = {
    "acne": "acne",
    "dark spots": "darkspots", 
    "wrinkles": "wrinkles"
}

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
        print(f"[Critical Error] Model not found at {MODEL_PATH}. Check your path!")
        exit()
        
    model = model.to(DEVICE)
    model.eval()
    return model

def process_multi_class():
    print(f"--- SCANNING & BALANCING DIRECTORY: Skin v2 ---")
    
    # 1. Setup Target Directories
    os.makedirs(EXTRAS_DIR, exist_ok=True)
    for final_folder in TARGET_CLASSES.values():
        os.makedirs(os.path.join(FINAL_DATASET_DIR, final_folder), exist_ok=True)

    if not os.path.exists(SOURCE_DIR):
        print(f"[Error] Could not find the source directory at: {SOURCE_DIR}")
        return

    # 2. Categorize Images by Folder Name
    print("Hunting for target classes in extracted files...")
    class_image_paths = {key: [] for key in TARGET_CLASSES.values()}
    
    for root, _, files in os.walk(SOURCE_DIR):
        if "__MACOSX" in root: continue # Skip junk files
        
        lower_root = root.lower()
        
        assigned_class = None
        for search_term, target_folder in TARGET_CLASSES.items():
            # Match folder names like "Dark Spots" to "darkspots"
            if search_term.replace(" ", "") in lower_root.replace(" ", ""):
                assigned_class = target_folder
                break
                
        if assigned_class:
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_image_paths[assigned_class].append(os.path.join(root, file))

    for cls, paths in class_image_paths.items():
        print(f" -> Found {len(paths)} raw images for '{cls}'.")

    # 3. Initialize AI
    model = load_ai_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. Process Each Class
    results_stats = {}

    for cls, paths in class_image_paths.items():
        if not paths:
            print(f"\nSkipping {cls}: No images found.")
            continue
            
        print(f"\n--- AI Sorting '{cls}' ---")
        dark_images, light_images = [], []
        
        for img_path in tqdm(paths, desc=f"Scoring {cls}"):
            try:
                pil_img = Image.open(img_path).convert('RGB')
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logit = model(input_tensor).item()
                
                if logit > 0:
                    dark_images.append(img_path)
                else:
                    light_images.append(img_path)
            except:
                pass 

        orig_dark = len(dark_images)
        orig_light = len(light_images)
        
        # Balance
        target_count = min(orig_dark, orig_light)
        print(f"[{cls}] Imbalance: {orig_dark} Dark vs {orig_light} Light. Balancing to {target_count} each.")
        
        if target_count == 0:
            print(f"[{cls}] Failed to balance (one category is empty).")
            continue
            
        selected_dark = set(random.sample(dark_images, target_count))
        selected_light = set(random.sample(light_images, target_count))
        
        # Copy files (safely, without deleting originals)
        dest_dir = os.path.join(FINAL_DATASET_DIR, cls)
        added_count = 0
        
        for img_list, tone in [(dark_images, "dark"), (light_images, "light")]:
            for img_path in img_list:
                safe_name = f"skinv2_{cls}_{tone}_{added_count}_{os.path.basename(img_path)}"
                
                if img_path in (selected_dark | selected_light):
                    shutil.copy(img_path, os.path.join(dest_dir, safe_name))
                else:
                    shutil.copy(img_path, os.path.join(EXTRAS_DIR, safe_name))
                added_count += 1
                
        results_stats[cls] = {
            "orig_dark": orig_dark, "orig_light": orig_light, "balanced_total": target_count * 2
        }

    # 5. Generate Master EDA Plot
    if results_stats:
        print("\nGenerating Consolidated EDA Dashboard...")
        sns.set_theme(style="whitegrid")
        classes_processed = list(results_stats.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("New Additions from 'Skin v2' Directory", fontsize=16, weight='bold')
        
        # Plot 1: Unbalanced
        dark_orig = [results_stats[c]["orig_dark"] for c in classes_processed]
        light_orig = [results_stats[c]["orig_light"] for c in classes_processed]
        
        x = np.arange(len(classes_processed))
        width = 0.35
        
        axes[0].bar(x - width/2, dark_orig, width, label='Dark Skin', color='#8D5524')
        axes[0].bar(x + width/2, light_orig, width, label='Light Skin', color='#FFC0CB')
        axes[0].set_title("Original Imbalance in Source")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes_processed)
        axes[0].set_ylabel("Image Count")
        axes[0].legend()
        
        # Plot 2: Balanced
        balanced_counts = [results_stats[c]["balanced_total"] // 2 for c in classes_processed]
        
        axes[1].bar(x - width/2, balanced_counts, width, label='Dark Skin', color='#8D5524')
        axes[1].bar(x + width/2, balanced_counts, width, label='Light Skin', color='#FFC0CB')
        axes[1].set_title("Safely Added to dataset_final")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes_processed)
        axes[1].legend()

        plot_path = os.path.join(FINAL_DATASET_DIR, "skinv2_folder_audit.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"[Saved] Consolidated dashboard saved to: {plot_path}")

    print("\n" + "="*40)
    print("DIRECTORY PROCESSING COMPLETE")
    print("="*40)
    for cls, stats in results_stats.items():
        print(f"{cls.ljust(12)}: Added {stats['balanced_total']} perfectly balanced images.")

if __name__ == "__main__":
    process_multi_class()