import os
from huggingface_hub import hf_hub_download

# Define your target path
target_dir = "datasets/quality/raw"

# Ensure the directory exists
os.makedirs(target_dir, exist_ok=True)

# Download the specific file to that directory
file_path = hf_hub_download(
    repo_id="chaofengc/IQA-PyTorch-Datasets",
    filename="gfiqa-20k.tgz",
    repo_type="dataset",
    local_dir=target_dir
)

print(f"Dataset downloaded to: {file_path}")
 