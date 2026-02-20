import os
import zipfile
import random
import io
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
ZIP_PATH = os.path.join(BASE_DIR, "CelebA.zip")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "binary_skin_tone_effnetb0.pth")

FINAL_DATASET_DIR = os.path.join(BASE_DIR, "dataset_final")
EXTRAS_DIR = os.path.join(FINAL_DATASET_DIR, "extras_unbalanced")

MAX_TO_SCAN = 4000  # We grab a 4000-image sample pool per class to avoid scanning all 200k
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

def process_celeba_surgical():
    print(f"--- SURGICAL ZIP EXTRACTION: CelebA ---")
    
    os.makedirs(EXTRAS_DIR, exist_ok=True)
    for folder in ["puffy_eyes", "clear_face"]:
        os.makedirs(os.path.join(FINAL_DATASET_DIR, folder), exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"[Error] Could not find CelebA zip at: {ZIP_PATH}")
        return

    # 1. Map the Zip Architecture In-Memory
    print("Mapping zip file structure...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        
        # Locate the attributes file
        attr_file_path = next((f for f in namelist if "list_attr_celeba.txt" in f), None)
        if not attr_file_path:
            print("[Critical Error] Could not find 'list_attr_celeba.txt' inside the zip!")
            return
            
        # Create a fast lookup dictionary mapping filename -> full zip path
        print("Creating image path index...")
        image_paths = {os.path.basename(f): f for f in namelist if f.lower().endswith(('.jpg', '.png', '.jpeg'))}

        # 2. Load and Filter Attributes
        print("Reading attributes directly from zip...")
        with zip_ref.open(attr_file_path) as attr_file:
            # skiprows=1 skips the total image count line to align headers properly
            df = pd.read_csv(attr_file, sep=r'\s+', skiprows=1)

        print("Filtering for physical traits...")
        # Puffy Eyes Filter
        puffy_candidates = df[df['Bags_Under_Eyes'] == 1].index.tolist()
        
        # Clear Face Filter
        clear_candidates = df[
            (df['Bags_Under_Eyes'] == -1) & 
            (df['Young'] == 1) & 
            (df['Heavy_Makeup'] == -1) & 
            (df['No_Beard'] == 1) & 
            (df['Eyeglasses'] == -1)
        ].index.tolist()

        random.shuffle(puffy_candidates)
        random.shuffle(clear_candidates)
        
        candidates = {
            "puffy_eyes": [f for f in puffy_candidates if f in image_paths][:MAX_TO_SCAN],
            "clear_face": [f for f in clear_candidates if f in image_paths][:MAX_TO_SCAN]
        }

        # 3. AI Initialization
        model = load_ai_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        results_stats = {}

        # 4. Surgical AI Scanning & Extraction
        for cls, filenames in candidates.items():
            print(f"\n--- AI Sorting '{cls}' ({len(filenames)} candidates) ---")
            dark_images, light_images = [], []
            
            for filename in tqdm(filenames, desc=f"Scoring {cls}"):
                zip_img_path = image_paths[filename]
                
                try:
                    # Read image bytes directly into RAM without saving to disk
                    img_bytes = zip_ref.read(zip_img_path)
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    
                    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        logit = model(input_tensor).item()
                    
                    if logit > 0:
                        dark_images.append(zip_img_path)
                    else:
                        light_images.append(zip_img_path)
                except Exception as e:
                    pass 

            orig_dark = len(dark_images)
            orig_light = len(light_images)
            
            # 5. Strict 1:1 Balancing
            target_count = min(orig_dark, orig_light)
            print(f"\n[{cls}] Found: {orig_dark} Dark vs {orig_light} Light. Forcing 1:1 balance at {target_count} each.")
            
            if target_count == 0:
                print(f"[{cls}] Failed to balance (one category is empty).")
                continue
                
            selected_dark = set(random.sample(dark_images, target_count))
            selected_light = set(random.sample(light_images, target_count))
            
            # 6. Physical Extraction of ONLY the balanced images
            dest_dir = os.path.join(FINAL_DATASET_DIR, cls)
            added_count = 0
            
            print(f"Extracting {target_count * 2} perfectly balanced images to disk...")
            for img_list, tone in [(dark_images, "dark"), (light_images, "light")]:
                for zip_img_path in img_list:
                    filename = os.path.basename(zip_img_path)
                    _, ext = os.path.splitext(filename)
                    safe_name = f"celeba_{cls}_{tone}_{added_count:05d}{ext.lower()}"
                    
                    final_path = os.path.join(dest_dir, safe_name) if zip_img_path in (selected_dark | selected_light) else os.path.join(EXTRAS_DIR, safe_name)
                    
                    # Write bytes straight from the zip to the hard drive
                    with open(final_path, 'wb') as f:
                        f.write(zip_ref.read(zip_img_path))
                        
                    added_count += 1
                    
            results_stats[cls] = {
                "orig_dark": orig_dark, "orig_light": orig_light, "balanced_total": target_count * 2
            }

    # 7. Generate EDA Dashboard
    if results_stats:
        print("\nGenerating CelebA EDA Dashboard...")
        sns.set_theme(style="whitegrid")
        classes_processed = list(results_stats.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("CelebA Surgical Extraction & AI Balancing", fontsize=16, weight='bold')
        
        dark_orig = [results_stats[c]["orig_dark"] for c in classes_processed]
        light_orig = [results_stats[c]["orig_light"] for c in classes_processed]
        
        x = np.arange(len(classes_processed))
        width = 0.35
        
        axes[0].bar(x - width/2, dark_orig, width, label='Dark Skin (Raw)', color='#8D5524')
        axes[0].bar(x + width/2, light_orig, width, label='Light Skin (Raw)', color='#FFC0CB')
        axes[0].set_title("Original Imbalance in CelebA Sample")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(classes_processed)
        axes[0].set_ylabel("Image Count")
        axes[0].legend()
        
        balanced_counts = [results_stats[c]["balanced_total"] // 2 for c in classes_processed]
        
        axes[1].bar(x - width/2, balanced_counts, width, label='Dark Skin', color='#8D5524')
        axes[1].bar(x + width/2, balanced_counts, width, label='Light Skin', color='#FFC0CB')
        axes[1].set_title("Safely Added to dataset_final")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(classes_processed)
        axes[1].legend()

        plot_path = os.path.join(FINAL_DATASET_DIR, "celeba_surgical_audit.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    print("\n" + "="*40)
    print("CELEBA SURGICAL EXTRACTION COMPLETE")
    print("="*40)
    for cls, stats in results_stats.items():
        print(f"{cls.ljust(12)}: Added {stats['balanced_total']} perfectly balanced images.")

if __name__ == "__main__":
    process_celeba_surgical()