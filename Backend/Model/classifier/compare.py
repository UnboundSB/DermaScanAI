import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import gc

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pointing to your target unseen images
TARGET_DIR = r"D:\Projects\DermaScanAI\datasets\processed\processed\gfiqa-224-png\image"

# Integrated your new final4 brain here
MODELS_TO_TEST = [
    os.path.join(SCRIPT_DIR, "symptom_classifier_final1.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final2.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final3.pth"),
    os.path.join(SCRIPT_DIR, "symptom_classifier_final4.pth")
]

CLASSES = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- INITIALIZING SEQUENTIAL VRAM-SAFE AUDIT ON {DEVICE.type.upper()} ---")
    
    if not os.path.exists(TARGET_DIR):
        print(f"[!] Critical Error: Target directory not found at {TARGET_DIR}")
        return

    # Inference-time transformations (Matches your production trainer)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("[!] No images found in the target directory.")
        return
        
    print(f"Found {len(image_files)} images for cross-model verification.")
    
    # Initialize results dictionary
    results_dict = {img: {"Image_File": img} for img in image_files}
    
    # --- SEQUENTIAL EXECUTION LOOP ---
    for weights_path in MODELS_TO_TEST:
        if not os.path.exists(weights_path):
            print(f"[!] Skipping missing model: {weights_path}")
            continue
            
        model_name = os.path.basename(weights_path).replace('.pth', '')
        print(f"\nEvaluating: {model_name}...")
        
        # 1. Load Architecture & Inject Weights
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # 2. Inference Run
        with torch.no_grad():
            for img_name in tqdm(image_files, desc=f"Running {model_name}"):
                img_path = os.path.join(TARGET_DIR, img_name)
                
                try:
                    # Open and normalize
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                    
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, 0)
                    
                    pred_class = CLASSES[pred_idx.item()]
                    confidence_pct = round(conf.item() * 100, 2)
                    
                    results_dict[img_name][f"{model_name}_Prediction"] = pred_class
                    results_dict[img_name][f"{model_name}_Confidence"] = confidence_pct
                except Exception:
                    continue

        # 3. Aggressive VRAM Purge (Keeping the pipeline clean)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # --- AGGREGATION & REPORTING ---
    df = pd.DataFrame(list(results_dict.values()))
    
    prediction_columns = [f"{os.path.basename(m).replace('.pth', '')}_Prediction" for m in MODELS_TO_TEST if os.path.exists(m)]
    
    # Consensus logic: Do all active models agree?
    def check_consensus(row):
        preds = [row[col] for col in prediction_columns if col in row and pd.notna(row[col])]
        return len(set(preds)) == 1 if len(preds) == len(prediction_columns) else False
        
    df['Total_Consensus'] = df.apply(check_consensus, axis=1)
    consensus_rate = (df['Total_Consensus'].sum() / len(df)) * 100
    
    print("\n" + "="*80)
    print(" CROSS-MODEL AUDIT COMPLETE ")
    print("="*80)
    print(f"Total Images Analyzed: {len(df)}")
    print(f"Full 4-Model Consensus Rate: {consensus_rate:.2f}%")
    
    print("\n--- SYMPTOM SENSITIVITY CHECK ---")
    for col in prediction_columns:
        m_label = col.replace('_Prediction', '')
        # Count how often this specific model flags 'wrinkles' or 'acne'
        wrinkle_pct = (df[col] == 'wrinkles').sum() / len(df) * 100
        acne_pct = (df[col] == 'acne').sum() / len(df) * 100
        print(f"{m_label:<20} | Wrinkles: {wrinkle_pct:>6.2f}% | Acne: {acne_pct:>6.2f}%")
        
    csv_path = os.path.join(SCRIPT_DIR, "cross_model_audit_results.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + "="*80)
    print(f"Comprehensive CSV generated: {csv_path}")

if __name__ == "__main__":
    main()