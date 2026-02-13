import os
import zipfile
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
# Adjusted target path for the Face Detection task
TARGET_DIR = r"D:\Projects\DermaScanAI\datasets\face_detection\raw"

# Ensure the directory exists
os.makedirs(TARGET_DIR, exist_ok=True)

print("--- DOWNLOADING WIDER FACE (RAW ZIPS) ---")

# 1. Download the Training Images (~1.47 GB)
print("Downloading WIDER_train.zip (This might take a moment)...")
train_zip_path = hf_hub_download(
    repo_id="CUHK-CSE/wider_face",
    filename="data/WIDER_train.zip",
    repo_type="dataset",
    local_dir=TARGET_DIR
)
print(f"[Success] Downloaded to: {train_zip_path}")

# 2. Download the Annotations (Bounding Boxes)
print("\nDownloading wider_face_split.zip (Annotations)...")
annot_zip_path = hf_hub_download(
    repo_id="CUHK-CSE/wider_face",
    filename="data/wider_face_split.zip",
    repo_type="dataset",
    local_dir=TARGET_DIR
)
print(f"[Success] Downloaded to: {annot_zip_path}")

# --- EXTRACTION ---
# The loader needs the physical folders, so we unzip them right here
print("\n--- EXTRACTING DATASETS ---")

train_extract_dir = os.path.join(TARGET_DIR, "WIDER_train")
if not os.path.exists(train_extract_dir):
    print("Extracting training images...")
    with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
        # Extracts to raw/WIDER_train/images/...
        zip_ref.extractall(TARGET_DIR)
    print(f"[Success] Images extracted to {train_extract_dir}")
else:
    print(f"[Info] Images already extracted at {train_extract_dir}")

split_extract_dir = os.path.join(TARGET_DIR, "wider_face_split")
if not os.path.exists(split_extract_dir):
    print("Extracting annotations...")
    with zipfile.ZipFile(annot_zip_path, 'r') as zip_ref:
        # Extracts to raw/wider_face_split/...
        zip_ref.extractall(TARGET_DIR)
    print(f"[Success] Annotations extracted to {split_extract_dir}")
else:
    print(f"[Info] Annotations already extracted at {split_extract_dir}")

print("\n[COMPLETE] Your raw Face Detection dataset is fully staged and ready!")