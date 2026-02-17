import os
import sys
import imagehash
from PIL import Image
from tqdm import tqdm

# --- 1. DYNAMIC IMPORT SETUP ---
# (Standard boilerplate to find your custom model)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.abspath(os.path.join(current_dir, "../../"))
if model_root not in sys.path:
    sys.path.append(model_root)

try:
    from quality.model import IQAModel
    print(f"[Init] IQAModel loaded successfully.")
except ImportError:
    print("[Error] Could not import IQAModel. Check your path!")
    sys.exit()

# --- CONFIGURATION ---
DATASET_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms\dataset_final"
CLASSES = ["clear_face", "darkspots", "puffy_eyes", "wrinkles", "pimples"]

# Duplicate Sensitivity (Lower = Stricter)
# 0 = Exact duplicate
# 1-3 = Very similar (slight compression/resize differences)
HASH_DIFF_THRESHOLD = 2 

def find_and_clean_duplicates():
    print(f"--- STARTING DUPLICATE REMOVAL ON {DATASET_DIR} ---")
    
    # 1. Initialize Quality Model
    iqa = IQAModel()
    
    total_removed = 0
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir): continue
        
        print(f"\nScanning '{cls}' for duplicates...")
        
        # Dictionary to store {hash: [list_of_files]}
        hashes = {}
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # --- STEP A: HASHING (Find the duplicates) ---
        for img_name in tqdm(images, desc="Hashing"):
            img_path = os.path.join(cls_dir, img_name)
            try:
                # Perceptual Hash (Resistant to resizing/compression)
                with Image.open(img_path) as img:
                    # hash_size=8 is standard
                    h = imagehash.phash(img, hash_size=8)
                    
                # Convert hash to string for storage
                h_str = str(h)
                
                if h_str in hashes:
                    hashes[h_str].append(img_path)
                else:
                    hashes[h_str] = [img_path]
                    
            except Exception as e:
                # Corrupt image? Skip it.
                continue

        # --- STEP B: RESOLVING (Keep best, delete rest) ---
        duplicates_groups = [files for files in hashes.values() if len(files) > 1]
        
        if not duplicates_groups:
            print(f"  -> No duplicates found in {cls}.")
            continue
            
        print(f"  -> Found {len(duplicates_groups)} sets of duplicates. resolving...")
        
        for file_group in tqdm(duplicates_groups, desc="Cleaning"):
            # If we have [img1.jpg, img2.jpg, img3.jpg]
            
            best_file = None
            best_score = -1.0
            
            # Score every file in the duplicate group
            for f_path in file_group:
                try:
                    score = iqa.predict(f_path)
                    # Unwrap tensor/list if necessary
                    if hasattr(score, 'item'): score = score.item()
                    if isinstance(score, list): score = score[0]
                    
                    if score > best_score:
                        best_score = score
                        best_file = f_path
                except:
                    # If scoring fails, treat score as 0
                    pass
            
            # If for some reason all failed, default to the first one
            if best_file is None:
                best_file = file_group[0]
            
            # Delete the losers
            for f_path in file_group:
                if f_path != best_file:
                    try:
                        os.remove(f_path)
                        total_removed += 1
                        # Optional: Print deleted file
                        # print(f"Deleted duplicate (Score {score:.2f}): {os.path.basename(f_path)}")
                    except OSError:
                        pass

    print("\n" + "="*40)
    print("DUPLICATE REMOVAL REPORT")
    print("="*40)
    print(f"Total Duplicate Images Removed: {total_removed}")
    print("Your dataset is now unique and optimized.")

if __name__ == "__main__":
    find_and_clean_duplicates()