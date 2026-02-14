import os
import shutil
import random
import yaml
from ultralytics import YOLO

# --- CONFIGURATION ---
# Source of our 640x640 processed data
SOURCE_DIR = r"D:\Projects\DermaScanAI\datasets\face_detection\processed_640"
IMAGES_DIR = os.path.join(SOURCE_DIR, "images")
LABELS_DIR = os.path.join(SOURCE_DIR, "labels")

# Where YOLO will organize the data for training
DATASET_ROOT = r"D:\Projects\DermaScanAI\datasets\face_detection\yolo_dataset"

# Training Settings
EPOCHS = 20           
BATCH_SIZE = 16       # 16 is safe for 6GB VRAM.
IMG_SIZE = 640
MODEL_NAME = "yolov8n.pt" # Nano model (Fastest)

# FINAL OUTPUT NAME
FINAL_MODEL_NAME = "face_detector.pt"

def organize_dataset():
    """
    Splits data into Train/Val and moves to YOLO structure.
    """
    # Check if we already did this to save time
    if os.path.exists(os.path.join(DATASET_ROOT, "train", "images")):
        print(f"[Info] Dataset found at {DATASET_ROOT}. Skipping organization.")
        return

    print("--- ORGANIZING DATASET (Train/Val Split) ---")
    
    # Get all label files (files with matching images)
    all_files = [f.replace(".txt", "") for f in os.listdir(LABELS_DIR) if f.endswith(".txt")]
    
    if not all_files:
        print("[Error] No label files found! Did preprocessing run?")
        return

    # Shuffle & Split 80/20
    random.seed(42)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total: {len(all_files)} | Train: {len(train_files)} | Val: {len(val_files)}")
    
    def move_split(file_list, split):
        img_dest = os.path.join(DATASET_ROOT, split, "images")
        lbl_dest = os.path.join(DATASET_ROOT, split, "labels")
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        for stem in file_list:
            # Copy Image
            src_img = os.path.join(IMAGES_DIR, stem + ".png")
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(img_dest, stem + ".png"))
            
            # Copy Label
            src_lbl = os.path.join(LABELS_DIR, stem + ".txt")
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, os.path.join(lbl_dest, stem + ".txt"))

    move_split(train_files, "train")
    move_split(val_files, "val")
    print("[Success] Dataset organized.")

def create_yaml_config():
    """Create data.yaml for YOLO."""
    yaml_path = os.path.join(DATASET_ROOT, "face_data.yaml")
    
    # YOLO needs absolute paths usually
    data = {
        'path': os.path.abspath(DATASET_ROOT),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['face']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return yaml_path

def train_and_rename(yaml_path):
    print(f"--- STARTING TRAINING ---")
    
    model = YOLO(MODEL_NAME)
    
    # 1. TRAIN
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,         # GPU
        workers=4,
        project="DermaScan_Runs", # Creates folder 'DermaScan_Runs'
        name="train_run",         # Creates subfolder 'train_run'
        exist_ok=True     # Overwrite if exists
    )
    
    # 2. LOCATE 'best.pt'
    # Ultralytics saves to: project/name/weights/best.pt
    best_weights_path = os.path.join("DermaScan_Runs", "train_run", "weights", "best.pt")
    
    # 3. RENAME & MOVE to Root
    if os.path.exists(best_weights_path):
        print(f"\n[Training Complete] Found weights at: {best_weights_path}")
        
        target_path = os.path.join(os.getcwd(), FINAL_MODEL_NAME)
        shutil.copy(best_weights_path, target_path)
        
        print("-" * 40)
        print(f"SUCCESS! Your model is ready: {target_path}")
        print("-" * 40)
    else:
        print("[Error] Could not find 'best.pt'. Training might have failed.")

if __name__ == "__main__":
    organize_dataset()
    config = create_yaml_config()
    train_and_rename(config)