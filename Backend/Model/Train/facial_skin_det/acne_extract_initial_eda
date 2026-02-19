import os
import zipfile
import shutil
import random
import cv2
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
ZIP_PATH = os.path.join(BASE_DIR, "acne.zip")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "binary_skin_tone_effnetb0.pth")

# Target Directories
FINAL_ACNE_DIR = os.path.join(BASE_DIR, "dataset_final", "acne")
EDA_DIR = os.path.join(FINAL_ACNE_DIR, "eda")
TEMP_EXTRACT_DIR = os.path.join(FINAL_ACNE_DIR, "_temp_extract")
EXTRAS_DIR = os.path.join(FINAL_ACNE_DIR, "extras_unbalanced")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

def load_ai_model():
    print(f"Loading EfficientNet-B0 on {DEVICE}...")
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1) # 1 Output Node
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"[Critical Error] Model not found at {MODEL_PATH}")
        exit()
        
    model = model.to(DEVICE)
    model.eval()
    return model

def get_brightness(image_path):
    """Calculates basic grayscale brightness for EDA comparison."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return np.mean(img) if img is not None else 0
    except:
        return 0

def process_and_balance():
    print(f"--- PROCESSING ACNE DATASET WITH AI SKIN DETECTOR ---")
    
    # 1. Setup
    for d in [FINAL_ACNE_DIR, EDA_DIR, TEMP_EXTRACT_DIR, EXTRAS_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"[Error] Could not find {ZIP_PATH}")
        return

    print("Extracting zip file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TEMP_EXTRACT_DIR)

    all_images = []
    for root, _, files in os.walk(TEMP_EXTRACT_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))

    print(f"Found {len(all_images)} images. Running AI inference...")

    # 2. AI Initialization
    model = load_ai_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Tracking lists
    dark_images, light_images = [], []
    dark_brightness, light_brightness = [], []
    ai_logits = []

    # 3. Scanning Loop
    for img_path in tqdm(all_images, desc="AI Sorting"):
        try:
            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                logit = model(input_tensor).item()
            
            ai_logits.append(logit)
            brightness = get_brightness(img_path)
            
            # Logit > 0 means Sigmoid prob > 0.5 (Dark Skin)
            if logit > 0:
                dark_images.append(img_path)
                dark_brightness.append(brightness)
            else:
                light_images.append(img_path)
                light_brightness.append(brightness)
                
        except Exception as e:
            os.remove(img_path) # Delete corrupt images immediately

    orig_dark_count = len(dark_images)
    orig_light_count = len(light_images)
    
    # 4. Strict Balancing
    target_count = min(orig_dark_count, orig_light_count)
    if target_count == 0:
        print("[Critical Error] One skin tone category is empty!")
        return

    print(f"\nBalancing Dataset to {target_count} images per category...")
    
    selected_dark = set(random.sample(dark_images, target_count))
    selected_light = set(random.sample(light_images, target_count))
    
    # 5. File Routing
    final_count = 0
    for img_list, category_name in [(dark_images, "dark"), (light_images, "light")]:
        for img_path in img_list:
            safe_name = f"{category_name}_{final_count}_{os.path.basename(img_path)}" 
            dest_folder = FINAL_ACNE_DIR if img_path in (selected_dark | selected_light) else EXTRAS_DIR
            shutil.move(img_path, os.path.join(dest_folder, safe_name))
            final_count += 1

    shutil.rmtree(TEMP_EXTRACT_DIR)

    # 6. Generate Master EDA Dashboard
    print("Generating EDA Dashboard...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Acne Dataset: AI Skin Tone Audit & Balancing Report", fontsize=16, weight='bold')

    # Plot 1: Unbalanced Source
    sns.barplot(x=["Dark Skin", "Light Skin"], y=[orig_dark_count, orig_light_count], 
                ax=axes[0,0], palette=["#8D5524", "#FFC0CB"])
    axes[0,0].set_title(f"Original Zip Content\nTotal: {orig_dark_count + orig_light_count}", fontsize=12)

    # Plot 2: Balanced Final
    sns.barplot(x=["Dark Skin", "Light Skin"], y=[target_count, target_count], 
                ax=axes[0,1], palette=["#8D5524", "#FFC0CB"])
    axes[0,1].set_title(f"Final Balanced Dataset\nTotal: {target_count * 2}", fontsize=12)

    # Plot 3: Brightness Distribution
    # This proves the AI isn't just thresholding brightness
    sns.kdeplot(dark_brightness, fill=True, color="#8D5524", label="Predicted Dark", ax=axes[1,0])
    sns.kdeplot(light_brightness, fill=True, color="#FFC0CB", label="Predicted Light", ax=axes[1,0])
    axes[1,0].set_title("Pixel Brightness vs. AI Prediction", fontsize=12)
    axes[1,0].set_xlabel("Average Brightness (0=Black, 255=White)")
    axes[1,0].legend()

    # Plot 4: AI Confidence/Logit Distribution
    sns.histplot(ai_logits, bins=40, kde=True, color="purple", ax=axes[1,1])
    axes[1,1].axvline(x=0, color='red', linestyle='--', label="Decision Boundary (0)")
    axes[1,1].set_title("AI Model Confidence (Logits)", fontsize=12)
    axes[1,1].set_xlabel("<- Light Skin | Dark Skin ->")
    axes[1,1].legend()

    plot_path = os.path.join(EDA_DIR, "acne_comprehensive_eda.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print("\n" + "="*40)
    print("ACNE DATASET FULLY PROCESSED")
    print("="*40)
    print(f"Kept (Balanced): {target_count * 2} images -> {FINAL_ACNE_DIR}")
    print(f"Discarded (Extras): {(orig_dark_count + orig_light_count) - (target_count * 2)} images -> {EXTRAS_DIR}")
    print(f"Dashboard Saved: {plot_path}")

if __name__ == "__main__":
    process_and_balance()