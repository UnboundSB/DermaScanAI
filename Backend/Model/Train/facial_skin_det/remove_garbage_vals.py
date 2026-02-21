import os
import shutil
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# --- CONFIGURATION ---
BASE_DIR = r"D:\Projects\DermaScanAI\datasets\skin_ageing_symptoms"
INPUT_DIR = os.path.join(BASE_DIR, "dataset_ready_for_training")

# We move the garbage here instead of deleting it permanently, just in case
TRASH_DIR = os.path.join(BASE_DIR, "deleted_garbage")

CLASSES = ["acne", "darkspots", "wrinkles", "puffy_eyes", "clear_face"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"--- STARTING MTCNN GARBAGE COLLECTION ON {DEVICE} ---")
    
    os.makedirs(TRASH_DIR, exist_ok=True)
    
    # Initialize MTCNN for strict face verification
    # keep_all=False ensures we just look for at least one highly probable face
    print("Loading Strict MTCNN Auditor...")
    mtcnn = MTCNN(keep_all=False, device=DEVICE)
    
    total_audited = 0
    total_trashed = 0
    
    for cls in CLASSES:
        source_folder = os.path.join(INPUT_DIR, cls)
        trash_folder = os.path.join(TRASH_DIR, cls)
        
        if not os.path.exists(source_folder):
            continue
            
        os.makedirs(trash_folder, exist_ok=True)
        images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images: continue
        
        print(f"\nAuditing '{cls}' ({len(images)} images)...")
        trashed_count = 0
        
        for img_name in tqdm(images):
            img_path = os.path.join(source_folder, img_name)
            
            try:
                # MTCNN requires RGB PIL image
                img = cv2.imread(img_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                # Detect face & get probability score
                boxes, probs = mtcnn.detect(pil_img)
                
                # STRICT FILTER: If no box found, or the AI is less than 90% sure it's a face
                if boxes is None or probs[0] < 0.90:
                    trash_path = os.path.join(trash_folder, img_name)
                    shutil.move(img_path, trash_path)
                    trashed_count += 1
                    
            except Exception as e:
                # If the image is completely corrupted and crashes the reader, toss it
                trash_path = os.path.join(trash_folder, img_name)
                shutil.move(img_path, trash_path)
                trashed_count += 1
                
        total_audited += len(images)
        total_trashed += trashed_count
        print(f"[{cls}] Purged {trashed_count} false positives.")

    print("\n" + "="*40)
    print("GARBAGE COLLECTION COMPLETE")
    print("="*40)
    print(f"Total Images Audited:  {total_audited}")
    print(f"Total Garbage Removed: {total_trashed}")
    print(f"Final Cleaned Dataset: {INPUT_DIR}")
    print(f"Garbage Safely Stored: {TRASH_DIR}")

if __name__ == "__main__":
    main()