import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
DATA_DIR = os.path.join(BASE_DIR, "dataset_augmented_224_png")
MODEL_PATH = r"D:\Projects\DermaScanAI\Backend\Model\classifier\symptom_classifier_final4.pth"
RESULTS_DIR = os.path.join(BASE_DIR, "Isolated_Testing_Results")

CLASSES = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_matrix(y_true, y_pred, title, filename):
    """Generates a locked 5x5 confusion matrix to map symptom leakage."""
    # Enforce all 5 labels so the matrix doesn't collapse on isolated classes
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.ylabel('Ground Truth (Actual Folder)', fontsize=12)
    plt.xlabel('AI Prediction', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300)
    plt.close()

def main():
    print(f"--- INITIATING CLASS-ISOLATED AUDIT ON {DEVICE.type.upper()} ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"[!] Error: Data directory not found at {DATA_DIR}")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. LOAD THE BRAIN
    print("Loading symptom_classifier_final4.pth...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 2. PRE-PROCESS PIPELINE
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    global_true = []
    global_pred = []

    # 3. ISOLATED FOLDER CRAWL
    for class_idx, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(folder_path):
            print(f"[!] Skipping missing folder: {class_name}")
            continue

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nScanning isolated class: [{class_name.upper()}] - {len(image_files)} images")

        class_true = []
        class_pred = []

        with torch.no_grad():
            for img_name in tqdm(image_files, desc=f"Evaluating {class_name}"):
                img_path = os.path.join(folder_path, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                    
                    outputs = model(input_tensor)
                    _, predicted_idx = torch.max(outputs, 1)
                    
                    pred_val = predicted_idx.item()
                    
                    class_true.append(class_idx)
                    class_pred.append(pred_val)
                    
                    global_true.append(class_idx)
                    global_pred.append(pred_val)
                except Exception:
                    continue
        
        # Plot the micro-matrix for this specific folder
        matrix_title = f"Isolated Leakage Map: {class_name.upper()}"
        matrix_filename = f"conf_matrix_{class_name}.png"
        plot_matrix(class_true, class_pred, matrix_title, matrix_filename)
        print(f"[*] Saved isolated micro-matrix: {matrix_filename}")

    # 4. THE FINAL VERDICT
    print("\n" + "="*80)
    print(" GENERATING GLOBAL AUDIT REPORT ")
    print("="*80)
    
    plot_matrix(global_true, global_pred, "Final Master Confusion Matrix (All Classes)", "finalconf_matrix.png")
    print(f"[*] Saved master matrix: finalconf_matrix.png")

    print("\n--- FINAL CLASSIFICATION REPORT ---")
    print(classification_report(global_true, global_pred, target_names=CLASSES))
    print(f"\nAudit Complete. All diagnostic reports are locked in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()