import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The exact folder containing the raw, unseen test images
TARGET_DIR = r"D:\Projects\DermaScanAI\datasets\processed\processed\gfiqa-224-png\image"

MODELS_TO_TEST = [
    os.path.join(SCRIPT_DIR, "symptom_classifier_final1.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final2.pth"), 
    os.path.join(SCRIPT_DIR, "symptom_classifier_final3.pth")
]

CLASSES = ['acne', 'clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path):
    if not os.path.exists(weights_path):
        print(f"[!] Critical Error: Missing {weights_path}")
        return None
        
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def main():
    print(f"--- INITIALIZING REAL-WORLD AUDIT ON {DEVICE.type.upper()} ---")
    
    if not os.path.exists(TARGET_DIR):
        print(f"[!] Error: Target directory not found at {TARGET_DIR}")
        return

    # Load all three models into a dictionary
    loaded_models = {}
    for path in MODELS_TO_TEST:
        name = os.path.basename(path).replace('.pth', '')
        loaded_models[name] = load_model(path)
        
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
    
    results = []
    
    # Run the gauntlet
    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(TARGET_DIR, img_name)
            
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            except Exception as e:
                print(f"Skipping {img_name} due to read error: {e}")
                continue
                
            row_data = {"Image_File": img_name}
            predictions = []
            
            for model_name, model in loaded_models.items():
                if model is None: continue
                
                outputs = model(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                
                pred_class = CLASSES[pred_idx.item()]
                confidence_pct = round(conf.item() * 100, 2)
                
                row_data[f"{model_name}_Prediction"] = pred_class
                row_data[f"{model_name}_Confidence"] = confidence_pct
                predictions.append(pred_class)
                
            # Check for consensus (Did all 3 models predict the exact same symptom?)
            consensus = len(set(predictions)) == 1
            row_data["Total_Consensus"] = consensus
            results.append(row_data)

    # --- GENERATE REPORTS ---
    df = pd.DataFrame(results)
    
    # Calculate how often the models agreed
    consensus_rate = (df['Total_Consensus'].sum() / len(df)) * 100
    
    # Calculate how often 'wrinkles' was predicted by each model
    print("\n" + "="*80)
    print(" UNSEEN DATASET AUDIT RESULTS ")
    print("="*80)
    print(f"Total Images Analyzed: {len(df)}")
    print(f"Complete Model Consensus Rate: {consensus_rate:.2f}%")
    print("\n--- 'WRINKLES' PREDICTION FREQUENCY ---")
    
    for model_name in loaded_models.keys():
        wrinkle_count = (df[f"{model_name}_Prediction"] == 'wrinkles').sum()
        wrinkle_pct = (wrinkle_count / len(df)) * 100
        print(f"{model_name}: {wrinkle_pct:.2f}% of all images flagged as wrinkles")
        
    csv_path = os.path.join(SCRIPT_DIR, "unseen_audit_results.csv")
    df.to_csv(csv_path, index=False)
    print("="*80)
    print(f"Detailed breakdown saved to: {csv_path}")

if __name__ == "__main__":
    main()