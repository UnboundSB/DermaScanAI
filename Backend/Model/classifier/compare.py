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

TARGET_DIR = r"D:\Projects\DermaScanAI\datasets\processed\processed\gfiqa-224-png\image"

MODELS_TO_TEST = [
    os.path.join(SCRIPT_DIR, "symptom_classifier_final1.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final2.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final3.pth")
]

CLASSES = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- INITIALIZING SEQUENTIAL VRAM-SAFE AUDIT ON {DEVICE.type.upper()} ---")
    
    if not os.path.exists(TARGET_DIR):
        print(f"[!] Critical Error: Target directory not found at {TARGET_DIR}")
        return

    # Standard inference transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("[!] No images found in the target directory.")
        return
        
    print(f"Found {len(image_files)} images for testing.")
    
    # Initialize a dictionary to hold results for every image
    results_dict = {img: {"Image_File": img} for img in image_files}
    
    # --- SEQUENTIAL EXECUTION ---
    for weights_path in MODELS_TO_TEST:
        if not os.path.exists(weights_path):
            print(f"[!] Skipping missing model: {weights_path}")
            continue
            
        model_name = os.path.basename(weights_path).replace('.pth', '')
        print(f"\nLoading and Evaluating: {model_name}...")
        
        # 1. Load Model into RAM/VRAM
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        # 2. Run Inference
        with torch.no_grad():
            for img_name in tqdm(image_files, desc=f"Processing with {model_name}"):
                img_path = os.path.join(TARGET_DIR, img_name)
                
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                except Exception as e:
                    continue
                    
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                
                pred_class = CLASSES[pred_idx.item()]
                confidence_pct = round(conf.item() * 100, 2)
                
                results_dict[img_name][f"{model_name}_Prediction"] = pred_class
                results_dict[img_name][f"{model_name}_Confidence"] = confidence_pct

        # 3. Aggressive Memory Cleanup (The CTO Move)
        print(f"Purging {model_name} from VRAM...")
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # --- GENERATE REPORTS ---
    # Convert dictionary to a flat list for pandas
    results_list = list(results_dict.values())
    df = pd.DataFrame(results_list)
    
    # Check consensus
    prediction_columns = [f"{os.path.basename(m).replace('.pth', '')}_Prediction" for m in MODELS_TO_TEST]
    
    def check_consensus(row):
        preds = [row[col] for col in prediction_columns if col in row and pd.notna(row[col])]
        return len(set(preds)) == 1 if preds else False
        
    df['Total_Consensus'] = df.apply(check_consensus, axis=1)
    
    consensus_rate = (df['Total_Consensus'].sum() / len(df)) * 100
    
    print("\n" + "="*80)
    print(" UNSEEN DATASET AUDIT RESULTS ")
    print("="*80)
    print(f"Total Images Analyzed: {len(df)}")
    print(f"Complete Model Consensus Rate: {consensus_rate:.2f}%")
    print("\n--- 'WRINKLES' PREDICTION FREQUENCY ---")
    
    for col in prediction_columns:
        if col in df.columns:
            wrinkle_count = (df[col] == 'wrinkles').sum()
            wrinkle_pct = (wrinkle_count / len(df)) * 100
            print(f"{col.replace('_Prediction', '')}: {wrinkle_pct:.2f}% of all images flagged as wrinkles")
        
    csv_path = os.path.join(SCRIPT_DIR, "unseen_audit_results.csv")
    df.to_csv(csv_path, index=False)
    print("="*80)
    print(f"Detailed breakdown saved to: {csv_path}")

if __name__ == "__main__":
    main()